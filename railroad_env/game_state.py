"""The rules engine - a faithful port of Game.java's turn resolution, scoring and disruption.

Turn order mirrors `Game.performGameUpdate` exactly:
    do_income -> compute_autobuilds -> do_actions (all PLACE_TRACKS, then all DISRUPT)
    -> do_instability_check (ink) -> score_connections -> compute_tile_states -> is_game_over
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C
from .grid import Coord, Grid, Tile, Town, get_rail_cost
from .pathfinding import autobuild, terrain_path_exists, train_bfs

# An action is a tuple: ("WAIT",) | ("PLACE", x, y) | ("DISRUPT", zone_id)
#                     | ("DISRUPT_AT", x, y) | ("AUTOPLACE", x1, y1, x2, y2)
Action = tuple


class GameState:
    # Observation channel layout (see get_observation).
    NUM_TOWN_CHANNELS = 12
    TOWN_CHANNEL_START = 10
    INSTABILITY_CHANNEL = TOWN_CHANNEL_START + NUM_TOWN_CHANNELS  # 22
    INKED_CHANNEL = INSTABILITY_CHANNEL + 1                       # 23
    ACTIVE_CONNECTION_CHANNEL = INKED_CHANNEL + 1                 # 24
    TOWNS_CHANNEL = ACTIVE_CONNECTION_CHANNEL + 1                 # 25
    SCORE_DIFF_CHANNEL = TOWNS_CHANNEL + 1                        # 26
    TURN_CHANNEL = SCORE_DIFF_CHANNEL + 1                         # 27
    NUM_CHANNELS = TURN_CHANNEL + 1                               # 28

    #: Divisor for the raw per-region track counts in channels 5 and 6.
    TRACK_COUNT_SCALE = 10.0

    #: Route-guide cell values (channels 10-21).
    ROUTE_ORIGIN_VALUE = -1.0     # the town this channel belongs to
    ROUTE_DEST_VALUE = -0.5       # a town it wants to reach
    ROUTE_PATH_VALUE = 1.0        # cells between them - the ones worth building on

    #: The two scalar channels are broadcast flat across the board. Their divisors are fixed
    #: rather than derived from max_turns, so a net trained on a short episode and evaluated on
    #: a full one reads the same scale for the same absolute turn.
    SCORE_DIFF_SCALE = 10_000.0
    TURN_SCALE = 100.0            # C.MAX_TURNS - the production game length

    def __init__(self, grid: Grid, max_turns: int = C.MAX_TURNS):
        self.grid = grid
        self.width = grid.width
        self.height = grid.height
        self.max_turns = max_turns

        self.scores = [0, 0]
        self.turn = 0
        self.instability_threshold = C.INSTABILITY_THRESHOLD_BASE

        self.paint_points = [0, 0]
        self.disruption_points = [0, 0]

        # Mirrors of the grid, kept in lockstep for cheap array access (encoding/rendering).
        self.tracks = np.full((grid.height, grid.width), C.TRACK_NONE, dtype=np.int32)
        self.terrain = np.zeros((grid.height, grid.width), dtype=np.int32)
        self.regions = np.zeros((grid.height, grid.width), dtype=np.int32)
        for (x, y), tile in grid.cells.items():
            self.terrain[y, x] = tile.type
            self.regions[y, x] = tile.zone_id

        self.town_positions: Dict[int, Coord] = {t.id: t.coord for t in grid.towns}
        self.town_position_set = set(self.town_positions.values())

        # Per-town open-terrain route guides only change when a zone gets inked.
        self._route_channels_cache: Optional[np.ndarray] = None

        self.do_income()

    # ------------------------------------------------------------------ accessors

    @property
    def towns(self) -> List[Town]:
        return self.grid.towns

    @property
    def zones(self):
        return self.grid.zones

    @property
    def inked_regions(self) -> set:
        return {z.id for z in self.grid.zones if z.inked}

    @property
    def region_instability(self) -> Dict[int, int]:
        return {z.id: z.instability for z in self.grid.zones}

    def get_terrain(self, x: int, y: int) -> int:
        tile = self.grid.get(x, y)
        return tile.type if tile is not None else -1

    def get_region(self, x: int, y: int) -> int:
        tile = self.grid.get(x, y)
        return tile.zone_id if tile is not None else -1

    def get_track_cost(self, x: int, y: int) -> int:
        tile = self.grid.get(x, y)
        return get_rail_cost(tile) if tile is not None else 0

    def is_done(self) -> bool:
        """Game.isGameOver: out of turns, or no desired connection is even theoretically
        reachable any more (every route between them severed by inked regions)."""
        return self.turn >= self.max_turns or not self._any_connection_still_possible()

    def _any_connection_still_possible(self) -> bool:
        for town in self.grid.towns:
            for other in town.desired_connections:
                if terrain_path_exists(self.grid, town, other):
                    return True
        return False

    def _is_placeable(self, x: int, y: int) -> bool:
        """Whether a track may legally be placed here right now, ignoring cost."""
        tile = self.grid.get(x, y)
        if tile is None:
            return False
        if tile.is_town():
            return False
        if tile.is_track():
            return False
        if self.grid.zones[tile.zone_id].inked:
            return False
        return True

    def can_disrupt(self, zone_id: int) -> bool:
        """Game.doActions' DISRUPT validation: the zone must exist, not be inked, and - the rule
        our earlier implementation was missing - must not contain a town."""
        if zone_id < 0 or zone_id >= len(self.grid.zones):
            return False
        zone = self.grid.zones[zone_id]
        return not zone.inked and not zone.contained_towns

    # ------------------------------------------------------------------ turn phases

    def do_income(self) -> None:
        self.paint_points = [C.PASSIVE_INCOME, C.PASSIVE_INCOME]
        self.disruption_points = [C.BLOT_POINTS_PER_TURN, C.BLOT_POINTS_PER_TURN]

    def _expand_autobuilds(self, actions: Sequence[Action]) -> List[Action]:
        """Game.computeAutobuilds: replace the (at most one) AUTOPLACE with the PLACE_TRACKS
        chain it resolves to. Extra AUTOPLACEs are dropped."""
        resolved: List[Action] = []
        autobuild_used = False
        for action in actions:
            if action and action[0] == "AUTOPLACE":
                if autobuild_used:
                    continue
                autobuild_used = True
                _, x1, y1, x2, y2 = action
                for coord in autobuild(self.grid, (x1, y1), (x2, y2)):
                    resolved.append(("PLACE", coord[0], coord[1], True))  # flagged autobuilt
            else:
                resolved.append(action)
        return resolved

    def resolve_turn(self, actions_per_player: Sequence[Sequence[Action]]) -> Dict:
        """Runs one full game turn for both players and returns a summary of what happened."""
        self.turn += 1

        expanded = [self._expand_autobuilds(actions) for actions in actions_per_player]
        placed, disrupted = self._do_actions(expanded)
        inked = self._do_instability_check()
        score_before = list(self.scores)
        self._score_connections()
        self._compute_tile_states()

        return {
            "placed": placed,
            "disrupted": disrupted,
            "inked": inked,
            "score_delta": [self.scores[i] - score_before[i] for i in range(2)],
        }

    def _do_actions(self, actions_per_player: Sequence[Sequence[Action]]):
        """Game.doActions: every player's PLACE_TRACKS resolve simultaneously against the
        turn-start board (a cell claimed by both becomes neutral, and both still pay), then
        every player's DISRUPT resolves."""
        tracks_placed: Dict[Coord, List[int]] = {}
        placement_order: List[Coord] = []

        for player_id, actions in enumerate(actions_per_player):
            interrupt_autobuild = False
            for action in actions:
                if not action or action[0] != "PLACE":
                    continue
                generated_by_autobuild = len(action) > 3 and action[3]
                if interrupt_autobuild and generated_by_autobuild:
                    continue

                x, y = action[1], action[2]
                tile = self.grid.get(x, y)
                if tile is None or tile.is_town():
                    continue
                if self.grid.zones[tile.zone_id].inked:
                    continue
                # Validated against the turn-start board: only the *same* player placing twice
                # on one cell is rejected here, so both players can claim the same free cell.
                if tile.is_track() or player_id in tracks_placed.get((x, y), ()):
                    continue

                rail_cost = get_rail_cost(tile)
                if self.paint_points[player_id] < rail_cost:
                    if generated_by_autobuild:
                        interrupt_autobuild = True
                    continue

                tracks_placed.setdefault((x, y), []).append(player_id)
                if (x, y) in placement_order:
                    placement_order.remove((x, y))
                placement_order.append((x, y))
                self.paint_points[player_id] -= rail_cost

        placed_per_player: List[List[Coord]] = [[], []]
        for coord in placement_order:
            player_idxs = tracks_placed[coord]
            owner = player_idxs[0] if len(player_idxs) == 1 else C.TRACK_NEUTRAL
            self._set_track(coord, owner)
            for pid in player_idxs:
                placed_per_player[pid].append(coord)

        disrupted_per_player: List[List[int]] = [[], []]
        for player_id, actions in enumerate(actions_per_player):
            for action in actions:
                if not action:
                    continue
                zone_id = None
                if action[0] == "DISRUPT":
                    zone_id = action[1]
                elif action[0] == "DISRUPT_AT":
                    tile = self.grid.get(action[1], action[2])
                    if tile is None:
                        continue
                    zone_id = tile.zone_id
                else:
                    continue

                if self.disruption_points[player_id] <= 0:
                    continue
                if not self.can_disrupt(zone_id):
                    continue

                self.disruption_points[player_id] -= 1
                self.grid.zones[zone_id].instability += 1
                disrupted_per_player[player_id].append(zone_id)

        return placed_per_player, disrupted_per_player

    def _set_track(self, coord: Coord, owner: int) -> None:
        self.grid.cells[coord].track = owner
        self.tracks[coord[1], coord[0]] = owner

    def _do_instability_check(self) -> List[int]:
        """Game.doInstabilityCheck: ink every zone at/over the threshold, wiping its tracks.
        Zones containing a town are immune (they also can't be disrupted in the first place)."""
        to_ink = [
            zone
            for zone in self.grid.zones
            if not zone.inked
            and not zone.contained_towns
            and zone.instability >= self.instability_threshold
        ]

        for zone in to_ink:
            self.instability_threshold += C.INSTABILITY_THRESHOLD_INCREASE
            zone.inked = True
            for coord in zone.coords:
                if self.grid.cells[coord].is_track():
                    self._set_track(coord, C.TRACK_NONE)
            self._route_channels_cache = None  # inked zones block open-terrain routes

        return [zone.id for zone in to_ink]

    def _score_connections(self) -> None:
        """Game.moveTrains: recompute every desired connection's shortest path and award each
        player 1 point per track they own on it. Neutral tracks and towns score for nobody."""
        for town in self.grid.towns:
            town.paths = {}
            town.active_connections = []
            for other in town.desired_connections:
                path = train_bfs(self.grid, town, other)
                if not path:
                    continue

                town.active_connections.append(other)
                town.paths[other.id] = path

                points_per_player = [0, 0]
                for coord in path:
                    track = self.grid.cells[coord].track
                    if track == 0 or track == 1:
                        points_per_player[track] += 1
                for player_id, points in enumerate(points_per_player):
                    if points:
                        self.scores[player_id] += points

    def _compute_tile_states(self) -> None:
        for tile in self.grid.cells.values():
            tile.active_connections.clear()
        for town in self.grid.towns:
            for to_town_id, path in town.paths.items():
                for coord in path:
                    self.grid.cells[coord].active_connections.append((town.id, to_town_id))

    def calculate_active_connections(self) -> Dict[Coord, set]:
        """{(x, y): {(from_town_id, to_town_id), ...}} for every cell on an active connection."""
        result: Dict[Coord, set] = defaultdict(set)
        for town in self.grid.towns:
            for to_town_id, path in town.paths.items():
                for coord in path:
                    result[coord].add((town.id, to_town_id))
        return result

    def find_connection(self, from_town_id: int, to_town_id: int) -> Optional[List[Coord]]:
        towns = {t.id: t for t in self.grid.towns}
        if from_town_id not in towns or to_town_id not in towns:
            return None
        path = train_bfs(self.grid, towns[from_town_id], towns[to_town_id])
        return path or None

    # ------------------------------------------------------------------ observation

    def _compute_route_channels(self) -> np.ndarray:
        """One channel per town id slot: the shortest *open-terrain* route (ignoring tracks,
        blocked only by inked zones) to each of that town's desired connections. Endpoints read
        1.0, interior cells 0.5 - a routing hint, not a statement of what is already built."""
        channels = np.zeros((self.height, self.width, self.NUM_TOWN_CHANNELS), dtype=np.float32)
        towns = {t.id: t for t in self.grid.towns}

        for town in self.grid.towns:
            if town.id >= self.NUM_TOWN_CHANNELS:
                continue
            for target in town.desired_connections:
                path = _open_terrain_route(self.grid, town, towns[target.id])
                if not path:
                    continue
                # Written in increasing precedence: route cells first, then the far town, then
                # this channel's own town. A cell can be several of these at once - a route may
                # run straight through a third town, which is passable - and the more specific
                # marker has to survive. Signs separate the two kinds of cell: the endpoints are
                # negative and the route between them positive, so a single 3x3 filter can tell
                # "somewhere to build" from "a town you must not build on".
                for (x, y) in path[1:-1]:
                    channels[y, x, town.id] = self.ROUTE_PATH_VALUE
                for (x, y) in (path[-1],):
                    channels[y, x, town.id] = self.ROUTE_DEST_VALUE
                channels[path[0][1], path[0][0], town.id] = self.ROUTE_ORIGIN_VALUE
        return channels

    def _get_route_channels(self) -> np.ndarray:
        if self._route_channels_cache is None:
            self._route_channels_cache = self._compute_route_channels()
        return self._route_channels_cache

    def get_observation(self) -> np.ndarray:
        """(height, width, NUM_CHANNELS) float32 view of the board from player 0's perspective.

        Channels:
          0-2    terrain one-hot (plains, river, mountain)
          3      enemy (player 1) track present
          4      own (player 0) track present
          5      enemy tracks in this cell's region, COUNT / 10 (not a density)
          6      own tracks in region, COUNT / 10 (neutral counts for both sides)
          7      region size / total map cells
          8      enemy tracks in region that sit on an active connection, COUNT / 10
          9      own tracks in region on an active connection, COUNT / 10
          10-21  per-town open-terrain route guides: -1 at this town, -0.5 at a town it
                 wants to reach, +1 on the cells between (see _compute_route_channels)
          22     region instability / threshold
          23     region inked
          24     cell is part of an active connection
          25     a town stands here (any town, ownerless and never buildable)
          26     (own score - enemy score) / 10000, the same value in every cell
          27     turn / 100, the same value in every cell
        """
        total_cells = float(self.width * self.height)
        obs = np.zeros((self.height, self.width, self.NUM_CHANNELS), dtype=np.float32)

        for t in range(3):
            obs[:, :, t] = self.terrain == t

        obs[:, :, 3] = self.tracks == 1
        obs[:, :, 4] = self.tracks == 0

        num_regions = len(self.grid.zones)
        active_mask = np.zeros((self.height, self.width), dtype=bool)
        for (x, y) in self.calculate_active_connections():
            active_mask[y, x] = True

        # Neutral (owner 2) tracks count toward both sides' regional presence.
        is_enemy_track = (self.tracks == 1) | (self.tracks == C.TRACK_NEUTRAL)
        is_own_track = (self.tracks == 0) | (self.tracks == C.TRACK_NEUTRAL)

        flat_regions = self.regions.ravel()
        enemy_by_region = np.bincount(flat_regions, weights=is_enemy_track.ravel(), minlength=num_regions)
        own_by_region = np.bincount(flat_regions, weights=is_own_track.ravel(), minlength=num_regions)
        size_by_region = np.bincount(flat_regions, minlength=num_regions).astype(np.float32)
        active_enemy = np.bincount(flat_regions, weights=(is_enemy_track & active_mask).ravel(), minlength=num_regions)
        active_own = np.bincount(flat_regions, weights=(is_own_track & active_mask).ravel(), minlength=num_regions)

        # RAW COUNTS, not densities. The quantity that decides whether a region gets inked is an
        # absolute difference - a disruptor compares enemy tracks against own tracks and acts on
        # a lead of 2 - so dividing by region size destroyed exactly the signal that matters.
        # Two enemy tracks in a 5-cell region and two in a 40-cell region read identically as a
        # density but are the same threat. Recovering the count from a density would have forced
        # the network to multiply channel 5 by channel 7 and by the board area, a product of
        # planes a convolution learns badly. TRACK_COUNT_SCALE is a fixed divisor, not a
        # normalizer: mean region size is ~8 cells, so these land in roughly 0..1.
        obs[:, :, 5] = (enemy_by_region / self.TRACK_COUNT_SCALE)[self.regions]
        obs[:, :, 6] = (own_by_region / self.TRACK_COUNT_SCALE)[self.regions]
        obs[:, :, 7] = size_by_region[self.regions] / total_cells
        # Counts too, for the same reason as 5 and 6: these are the tracks actually being PAID
        # for each turn, so the absolute number is the score rate, while a fraction of region
        # size is not a quantity the game ever uses.
        obs[:, :, 8] = (active_enemy / self.TRACK_COUNT_SCALE)[self.regions]
        obs[:, :, 9] = (active_own / self.TRACK_COUNT_SCALE)[self.regions]

        obs[:, :, self.TOWN_CHANNEL_START:self.TOWN_CHANNEL_START + self.NUM_TOWN_CHANNELS] = (
            self._get_route_channels()
        )

        instability_by_region = np.zeros(num_regions, dtype=np.float32)
        inked_by_region = np.zeros(num_regions, dtype=np.float32)
        for zone in self.grid.zones:
            instability_by_region[zone.id] = zone.instability
            inked_by_region[zone.id] = 1.0 if zone.inked else 0.0

        obs[:, :, self.INSTABILITY_CHANNEL] = (
            instability_by_region[self.regions] / C.INSTABILITY_THRESHOLD_BASE
        )
        obs[:, :, self.INKED_CHANNEL] = inked_by_region[self.regions]
        obs[:, :, self.ACTIVE_CONNECTION_CHANNEL] = active_mask

        for town in self.grid.towns:
            obs[town.coord[1], town.coord[0], self.TOWNS_CHANNEL] = 1.0

        # Flat planes. A convolution has no other way to see a board-wide scalar: every filter
        # is local, so the only way the score gap and the clock reach the policy is by being
        # present at every cell it looks at.
        obs[:, :, self.SCORE_DIFF_CHANNEL] = (
            (self.scores[0] - self.scores[1]) / self.SCORE_DIFF_SCALE
        )
        obs[:, :, self.TURN_CHANNEL] = self.turn / self.TURN_SCALE

        return obs


def _open_terrain_route(grid: Grid, from_town: Town, to_town: Town) -> List[Coord]:
    """Shortest route ignoring tracks and terrain type, blocked only by inked zones - the
    routing guide behind observation channels 10-21. Same N/E/S/W tie-break as train_bfs."""
    from collections import deque

    start, goal = from_town.coord, to_town.coord
    if start == goal:
        return [start]

    fifo = deque([start])
    visited = {start}
    prev: Dict[Coord, Optional[Coord]] = {start: None}

    while fifo:
        current = fifo.popleft()
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.insert(0, node)
                node = prev[node]
            return path
        for neigh in grid.get_neighbours(current):
            if neigh in visited:
                continue
            if grid.zones[grid.cells[neigh].zone_id].inked:
                continue
            visited.add(neigh)
            prev[neigh] = current
            fifo.append(neigh)
    return []
