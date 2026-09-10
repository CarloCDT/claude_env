"""Faithful port of the Java referee's GridMaker: mountains, rivers, zones, towns, connections.

The structure and ordering here deliberately mirror `GridMaker.java` step for step, including
its quirks (e.g. the fallback town-placement pass that skips the terrain/edge checks, and the
sequential reciprocal-connection filter whose result depends on town order). Random *streams*
differ from Java's - we use numpy - but every distribution and range matches.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import constants as C
from .grid import Coord, Grid, Town, Zone, manhattan


class JavaRandom:
    """Thin numpy wrapper exposing java.util.Random's method names/semantics, so the port below
    reads like the original. Streams are not bit-identical to Java's, only distributionally."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    def next_int(self, a: int, b: Optional[int] = None) -> int:
        """next_int(n) -> [0, n); next_int(a, b) -> [a, b)."""
        if b is None:
            return int(self._rng.randint(a))
        return int(self._rng.randint(a, b))

    def next_float(self, bound: float = 1.0) -> float:
        """[0, bound)."""
        return float(self._rng.random_sample() * bound)

    def next_boolean(self) -> bool:
        return bool(self._rng.randint(2))

    def shuffle(self, items: list) -> None:
        self._rng.shuffle(items)

    def choice(self, items: list):
        return items[self.next_int(len(items))]


class _River:
    __slots__ = ("current", "history", "preferred_direction")

    def __init__(self, coord: Coord, history: List[Coord], preferred_direction: Tuple[int, int]):
        self.current = coord
        self.history = list(history)
        self.history.append(coord)
        self.preferred_direction = preferred_direction

    def is_start(self) -> bool:
        return len(self.history) == 1


def _opposite(direction: Tuple[int, int]) -> Tuple[int, int]:
    return (-direction[0], -direction[1])


class GridMaker:
    def __init__(self, random: JavaRandom, height: Optional[int] = None):
        """`height` pins the grid height instead of drawing it randomly (width is still derived
        from the aspect ratio, exactly as the original does)."""
        self.random = random
        if height is None:
            self.h = random.next_int(C.MIN_GRID_HEIGHT, C.MAX_GRID_HEIGHT + 1)
        else:
            self.h = height
        # Java's Math.round on a float: floor(x + 0.5), so 22.5 -> 23 (not banker's rounding).
        self.w = int(math.floor(self.h * C.ASPECT_RATIO + 0.5))

        self.grid: Grid = None
        self.free_borders: List[Coord] = []
        self.free_coords: List[Coord] = []

    # ---------- helpers ----------

    def _is_corner(self, x: int, y: int) -> bool:
        return (
            (x == 0 and y == 0)
            or (x == 0 and y == self.h - 1)
            or (x == self.w - 1 and y == 0)
            or (x == self.w - 1 and y == self.h - 1)
        )

    def _is_edge(self, x: int, y: int) -> bool:
        return x == 0 or y == 0 or x == self.w - 1 or y == self.h - 1

    def _is_accessible(self, tile) -> bool:
        return not tile.is_water() and tile.type != C.TYPE_MOUNTAIN

    def _has_water_nearby(self, coord: Coord, river_history: List[Coord]) -> bool:
        ignore = river_history[max(len(river_history) - 2, 0):]
        for n in self.grid.get_neighbours(coord, C.ADJACENCY_8):
            if n in ignore:
                continue
            if self.grid.cells[n].is_water():
                return True
        return False

    def _available_neighbours(self, coords, predicate) -> List[Coord]:
        """Deduplicated union of the neighbours of `coords` passing `predicate` (the Java uses a
        HashSet here; only set membership matters since callers pick uniformly at random)."""
        available: Dict[Coord, None] = {}
        for coord in coords:
            for n in self.grid.get_neighbours(coord):
                if predicate(self.grid.cells[n]):
                    available[n] = None
        return list(available)

    # ---------- generation ----------

    def make(self) -> Grid:
        self._initialize_grid()
        self._make_mountains()
        self._make_rivers()

        average_tiles_per_zone = self.h // C.AVERAGE_TILES_PER_ZONE_COEFF_TO_GRID_HEIGHT
        n_zones = max(1, (self.h * self.w) // average_tiles_per_zone)
        zones = self._make_zones(n_zones)

        n_towns = max(4, (self.h * self.w) // C.AVERAGE_TILES_PER_TOWN)
        towns = self._make_towns(zones, n_towns, average_tiles_per_zone)

        self._make_town_connections(towns)

        self.grid.towns = towns
        self.grid.zones = zones
        return self.grid

    def _initialize_grid(self) -> None:
        self.grid = Grid(self.w, self.h)
        free_borders = []
        for y in range(self.h):
            for x in range(self.w):
                self.grid.cells[(x, y)].type = C.TYPE_GRASS
                if self._is_edge(x, y) and not self._is_corner(x, y):
                    free_borders.append((x, y))

        free_coords = list(self.grid.cells.keys())
        self.random.shuffle(free_borders)
        self.random.shuffle(free_coords)
        self.free_borders = free_borders
        self.free_coords = free_coords

    def _make_mountains(self) -> None:
        n_mountains = max(
            C.MIN_MOUNTAINS,
            int(math.floor(self.random.next_float(self.w * self.h * C.MOUNTAIN_TO_CELL_RATIO) + 0.5)),
        )
        if self.w * self.h < 10:
            n_mountains = 0

        for _ in range(n_mountains):
            mountain_size = self.random.next_int(2, 8)
            if not self.free_coords:
                return
            mountain_base = self.free_coords.pop(0)

            mountain_coords = [mountain_base]
            self.grid.cells[mountain_base].type = C.TYPE_MOUNTAIN
            if mountain_base in self.free_borders:
                self.free_borders.remove(mountain_base)

            for _ in range(mountain_size):
                neighs = self._available_neighbours(mountain_coords, self._is_accessible)
                if not neighs:
                    break
                new_mountain = self.random.choice(neighs)
                self.grid.cells[new_mountain].type = C.TYPE_MOUNTAIN
                mountain_coords.append(new_mountain)
                if new_mountain in self.free_borders:
                    self.free_borders.remove(new_mountain)
                if new_mountain in self.free_coords:
                    self.free_coords.remove(new_mountain)

    def _direction_from_river_start(self, start: Coord) -> Tuple[int, int]:
        x, y = start
        if x == 0:
            return C.EAST
        if y == 0:
            return C.SOUTH
        if x == self.w - 1:
            return C.WEST
        if y == self.h - 1:
            return C.NORTH
        return (0, 0)

    def _create_water(self, rivers_to_expand: List[_River], river: _River) -> None:
        rivers_to_expand.append(river)
        self.grid.cells[river.current].type = C.TYPE_WATER
        if river.current in self.free_coords:
            self.free_coords.remove(river.current)

    def _neighbours_for_river_to_flow(self, river: _River) -> List[Coord]:
        result = []
        for n in self.grid.get_neighbours(river.current):
            tile = self.grid.cells[n]
            if self._has_water_nearby(n, river.history):
                continue
            if tile.is_water():
                continue
            if river.is_start() and self._is_edge(n[0], n[1]):
                continue
            if not self._is_accessible(tile):
                continue
            result.append(n)
        return result

    @staticmethod
    def _weight(neigh: Coord, current: Coord, preferred: Tuple[int, int]) -> float:
        direction = (neigh[0] - current[0], neigh[1] - current[1])
        if direction == preferred:
            return 1.75
        if direction == _opposite(preferred):
            return 0.25
        return 1.0

    def _random_weighted_coord(self, weights: Dict[Coord, float]) -> Coord:
        total = sum(weights.values())
        rand = self.random.next_float(total)
        cumulative = 0.0
        for coord, weight in weights.items():
            cumulative += weight
            if rand < cumulative:
                del weights[coord]
                return coord
        return next(iter(weights))

    def _make_rivers(self) -> None:
        n_river_cells = int(math.floor(self.w * self.h * C.RIVER_TO_LAND_MIN_RATIO + 0.5))

        available_sources = list(self.free_borders)
        if self.random.next_boolean():
            available_sources.insert(0, (self.w // 2, self.h // 2))
        else:
            available_sources.insert(0, (self.w // 2, self.h // 2 + 1))

        initial_river = True
        generated_rivers: List[_River] = []

        while n_river_cells > 0 and available_sources:
            river_start = available_sources.pop(0)
            if river_start in self.free_borders:
                self.free_borders.remove(river_start)

            direction = self._direction_from_river_start(river_start)

            if self._has_water_nearby(river_start, []):
                continue

            rivers_to_expand: List[_River] = []
            self._create_water(rivers_to_expand, _River(river_start, [], direction))
            n_river_cells -= 1

            while rivers_to_expand:
                river = rivers_to_expand.pop(0)
                current = river.current

                if self._is_edge(current[0], current[1]) and not river.is_start():
                    generated_rivers.append(river)
                    continue

                neighs = self._neighbours_for_river_to_flow(river)
                if not neighs:
                    generated_rivers.append(river)
                    continue

                going_to_split = (
                    n_river_cells > 0
                    and len(neighs) >= 2
                    and self.random.next_float() <= C.RIVER_SPLIT_PROBA
                )
                going_to_split = going_to_split or initial_river
                initial_river = False

                weights = {n: self._weight(n, current, river.preferred_direction) for n in neighs}
                next_coord = self._random_weighted_coord(weights)

                next_direction = (
                    (next_coord[0] - current[0], next_coord[1] - current[1])
                    if going_to_split
                    else river.preferred_direction
                )
                next_river = _River(next_coord, river.history, next_direction)
                self._create_water(rivers_to_expand, next_river)
                n_river_cells -= 1

                if not going_to_split:
                    continue

                remaining = [c for c in neighs if c != next_coord]
                self.random.shuffle(remaining)
                if not remaining:
                    continue

                split_coord = remaining[0]
                split_direction = (split_coord[0] - current[0], split_coord[1] - current[1])
                self._create_water(rivers_to_expand, _River(split_coord, river.history, split_direction))
                n_river_cells -= 1

        self._delete_short_rivers(generated_rivers)

    def _delete_short_rivers(self, generated_rivers: List[_River]) -> None:
        for river in generated_rivers:
            if len(river.history) >= C.MIN_RIVER_LENGTH:
                continue
            for coord in river.history:
                self.grid.cells[coord].type = C.TYPE_GRASS
                if coord not in self.free_coords:
                    self.free_coords.append(coord)

    def _make_zones(self, n_zones: int) -> List[Zone]:
        cols = int(math.ceil(math.sqrt(n_zones)))
        rows = int(math.ceil(n_zones / cols))
        cell_h = self.h / rows
        zones: List[Zone] = []

        for i in range(n_zones):
            row = i // cols
            col = i % cols
            if row == rows - 1 and n_zones % cols != 0:
                last_row_points = n_zones % cols
                cell_w = self.w / last_row_points
            else:
                cell_w = self.w / cols
            x = (col + 0.5) * cell_w
            y = (row + 0.5) * cell_h
            zone_center = (int(x), int(y))
            zones.append(Zone(i, [zone_center]))
            self.grid.cells[zone_center].zone_id = i

        blocked_zones = set()
        zone_id = 0
        while True:
            if zone_id not in blocked_zones:
                zone = zones[zone_id]
                neighs = self._available_neighbours(zone.coords, lambda t: t.zone_id == -1)
                if not neighs:
                    blocked_zones.add(zone_id)
                    if len(blocked_zones) == n_zones:
                        break
                else:
                    neigh_to_add = self.random.choice(neighs)
                    self.grid.cells[neigh_to_add].zone_id = zone_id
                    zone.coords.append(neigh_to_add)
            zone_id = (zone_id + 1) % n_zones

        for zone in zones:
            all_neighs = set()
            for coord in zone.coords:
                for neigh in self.grid.get_neighbours(coord):
                    other_id = self.grid.cells[neigh].zone_id
                    # `>= 0` guards the (unreachable in practice) case of a cell the growth
                    # loop never claimed - without it a -1 would index the last zone.
                    if other_id != zone.id and other_id >= 0:
                        all_neighs.add(other_id)
            zone.neighbours = sorted(all_neighs)

        return zones

    def _make_towns(self, zones: List[Zone], n_towns: int, average_tiles_per_zone: int) -> List[Town]:
        available_zones = list(zones)
        blacklist: List[Zone] = []
        towns: List[Town] = []

        step = max(1, C.AVERAGE_TILES_PER_TOWN // average_tiles_per_zone)
        retries = 100
        if not available_zones:
            return towns

        i = 0
        while i < n_towns:
            index = self.random.next_int(i * step, (i + 1) * step)
            zone = available_zones[index % len(available_zones)]
            found = False

            if zone not in blacklist:
                town_coords = list(zone.coords)
                self.random.shuffle(town_coords)

                for town_coord in town_coords:
                    if (
                        self._is_accessible(self.grid.cells[town_coord])
                        and not self._is_edge(town_coord[0], town_coord[1])
                        and all(manhattan(town_coord, t.coord) >= C.MIN_TOWN_DISTANCE for t in towns)
                    ):
                        towns.append(Town(i, town_coord))
                        found = True
                        blacklist.append(zone)
                        blacklist.extend(zones[zid] for zid in zone.neighbours)
                        break

            if not found and retries > 0:
                retries -= 1
                continue  # retry the same index i (the Java's `i--` followed by the loop's `i++`)
            i += 1

        # Fallback pass: fill any towns the zone-based placement couldn't. Note this deliberately
        # skips the terrain/edge checks (as the original does) - the town tile is forced to plains
        # at the end regardless.
        towns_left_to_place = n_towns - len(towns)
        blacklist_ids = {id(z) for z in blacklist}
        available_zones = [z for z in available_zones if id(z) not in blacklist_ids]
        self.random.shuffle(available_zones)

        for i in range(towns_left_to_place):
            if not available_zones:
                break
            z = available_zones.pop(0)
            town_coords = list(z.coords)
            self.random.shuffle(town_coords)
            for town_coord in town_coords:
                if all(manhattan(town_coord, t.coord) >= C.MIN_TOWN_DISTANCE for t in towns):
                    towns.append(Town(i, town_coord))
                    break

        for idx, town in enumerate(towns):
            town.id = idx

        for town in towns:
            tile = self.grid.cells[town.coord]
            zones[tile.zone_id].contained_towns.append(town)
            tile.town_id = town.id
            tile.type = C.TYPE_GRASS

        return towns

    def _make_town_connections(self, towns: List[Town]) -> None:
        for t in towns:
            other_towns = [ot for ot in towns if ot is not t]
            self.random.shuffle(other_towns)
            at_least = min(3, len(other_towns))
            at_most = max(at_least, len(other_towns) - 4)
            count = self.random.next_int(at_least, at_most + 1)
            t.desired_connections = sorted(other_towns[:count], key=lambda town: town.id)

        # Remove reciprocal connections so every desired connection is unilateral. This is
        # sequential and order-dependent in the original: later towns are filtered against the
        # ALREADY-filtered lists of earlier ones.
        for t in towns:
            t.desired_connections = [
                ot for ot in t.desired_connections if t not in ot.desired_connections
            ]
