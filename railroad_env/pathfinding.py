"""Pathfinding ported from the Java referee: TrainBFS, TerrainAStar, AutobuildAStar."""
import heapq
import itertools
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from . import constants as C
from .grid import Coord, Grid, Town, get_rail_cost, manhattan


def train_bfs(grid: Grid, from_town: Town, to_town: Town) -> List[Coord]:
    """The active-connection path: shortest route from `from_town` to `to_town` through cells
    that already hold a track (any owner) or are towns. Returns [] if no path exists.

    Neighbours are expanded in N, E, S, W order with `visited` marked at enqueue time, which is
    what implements the statement's tie-break rule ("prioritize NORTH, EAST, SOUTH, WEST when
    moving from the requesting town to the desired connected town"). The Java sorts neighbours
    with a comparator that always returns a non-negative value, so its stable sort leaves
    Grid.ADJACENCY's N/E/S/W order untouched - this reproduces that.
    """
    start = from_town.coord
    goal = to_town.coord

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

        for neighbour in grid.get_neighbours(current):
            if grid.can_train_pass(neighbour) and neighbour not in visited:
                visited.add(neighbour)
                prev[neighbour] = current
                fifo.append(neighbour)

    return []


def terrain_path_exists(grid: Grid, from_town: Town, to_town: Town) -> bool:
    """TerrainAStar: is a connection between these towns still *theoretically* possible? Ignores
    tracks and terrain cost entirely - only inked zones block. Used for the "no connection is
    still possible" game-over condition."""
    start = from_town.coord
    goal = to_town.coord
    if start == goal:
        return True

    counter = itertools.count()
    open_heap = [(manhattan(start, goal), next(counter), start)]
    g_scores = {start: 0}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return True

        for neigh in grid.get_neighbours(current):
            if grid.zones[grid.cells[neigh].zone_id].inked:
                continue
            tentative_g = g_scores[current] + 1
            if tentative_g < g_scores.get(neigh, float("inf")):
                g_scores[neigh] = tentative_g
                heapq.heappush(open_heap, (tentative_g + manhattan(neigh, goal), next(counter), neigh))

    return False


class _AutobuildState:
    """Mirrors AutobuildState: identity (and therefore A* dedup) is (cursor, cursor_state)."""

    __slots__ = ("coord_to_build", "cursor", "cursor_state", "money_spent", "tracks")

    BUILDABLE = "BUILDABLE"
    BUILT = "BUILT"
    BROKEN = "BROKEN"

    def __init__(self, cursor, cursor_state, money_spent=0, tracks=frozenset(), coord_to_build=None):
        self.cursor = cursor
        self.cursor_state = cursor_state
        self.money_spent = money_spent
        self.tracks = tracks
        # The PLACE_TRACKS coord this step generates, or None if the cursor just moved.
        self.coord_to_build = coord_to_build

    def key(self):
        return (self.cursor, self.cursor_state)


def _is_part_of_rail_block(grid: Grid, coord: Coord, goal: Coord) -> bool:
    """Is `goal` reachable from `coord` walking only over existing tracks/towns?"""
    fifo = deque([coord])
    visited = {coord}
    while fifo:
        current = fifo.popleft()
        if current == goal:
            return True
        for n in grid.get_neighbours(current):
            if n not in visited and grid.cells[n].is_track_or_town():
                fifo.append(n)
                visited.add(n)
    return False


def _all_neighbours_of_rail_block(grid: Grid, cursor: Coord) -> List[Coord]:
    """Every buildable cell touching the contiguous track/town block the cursor sits on."""
    neighs: List[Coord] = []
    fifo = deque([cursor])
    visited = {cursor}
    while fifo:
        current = fifo.popleft()
        if not grid.cells[current].is_track_or_town():
            neighs.append(current)
            continue
        for n in grid.get_neighbours(current):
            if n not in visited:
                fifo.append(n)
                visited.add(n)
    return neighs


def autobuild(grid: Grid, from_coord: Coord, to_coord: Coord) -> List[Coord]:
    """AUTOPLACE: the cheapest chain of track placements linking `from_coord` to `to_coord`,
    returned as the list of cells to PLACE_TRACKS (in order). Empty if impossible or if the two
    are already connected by existing track.

    Note the original's heuristic is `cursor.manhattanTo(cursor)` - i.e. always 0 - so this is a
    uniform-cost search, not a guided A*. Reproduced as-is: with an admissible heuristic the
    generated paths would differ from the real game's.
    """
    if from_coord not in grid.cells or to_coord not in grid.cells:
        return []

    from_tile = grid.cells[from_coord]
    start = _AutobuildState(
        cursor=from_coord,
        cursor_state=_AutobuildState.BUILT if from_tile.is_track_or_town() else _AutobuildState.BUILDABLE,
    )

    def is_goal(state: _AutobuildState) -> bool:
        if state.cursor == to_coord:
            return True
        if grid.cells[state.cursor].is_track_or_town():
            return _is_part_of_rail_block(grid, state.cursor, to_coord)
        return False

    def successors(state: _AutobuildState) -> List[_AutobuildState]:
        if state.cursor_state == _AutobuildState.BROKEN:
            return []

        scope: List[Coord] = []
        current_tile = grid.cells[state.cursor]
        usable_track = state.cursor in state.tracks or current_tile.is_track_or_town()

        if usable_track:
            if state.cursor in state.tracks:
                scope.extend(grid.get_neighbours(state.cursor))
            else:
                scope.extend(_all_neighbours_of_rail_block(grid, state.cursor))
        else:
            scope.append(state.cursor)

        result: List[_AutobuildState] = []
        seen = set()
        for neigh in scope:
            tile = grid.cells[neigh]
            zone = grid.zones[tile.zone_id]
            if not (not zone.inked or tile.is_town()):
                continue
            if neigh in state.tracks:
                continue
            if tile.is_track_or_town():
                if neigh == state.cursor:
                    continue
                nxt = _AutobuildState(neigh, _AutobuildState.BUILT, state.money_spent, state.tracks)
            else:
                nxt = _AutobuildState(
                    neigh,
                    _AutobuildState.BUILT,
                    state.money_spent + get_rail_cost(tile),
                    state.tracks | {neigh},
                    coord_to_build=neigh,
                )
            if nxt.key() in seen:
                continue
            seen.add(nxt.key())
            result.append(nxt)
        return result

    counter = itertools.count()
    open_heap = [(0.0, 0, next(counter), start)]
    g_scores = {start.key(): 0.0}
    came_from: Dict[Tuple, _AutobuildState] = {}
    states: Dict[Tuple, _AutobuildState] = {start.key(): start}

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if is_goal(current):
            chain: List[Coord] = []
            node = current
            while node is not None:
                if node.coord_to_build is not None:
                    chain.insert(0, node.coord_to_build)
                node = came_from.get(node.key())
            return chain

        for neighbour in successors(current):
            tentative_g = g_scores[current.key()] + (neighbour.money_spent - current.money_spent)
            if tentative_g < g_scores.get(neighbour.key(), float("inf")):
                came_from[neighbour.key()] = current
                g_scores[neighbour.key()] = tentative_g
                states[neighbour.key()] = neighbour
                # Tie-break by the direction taken, matching AutobuildAStar.tieBreaker.
                delta = (neighbour.cursor[0] - current.cursor[0], neighbour.cursor[1] - current.cursor[1])
                tie = C.ADJACENCY.index(delta) if delta in C.ADJACENCY else len(C.ADJACENCY)
                heapq.heappush(open_heap, (tentative_g, tie, next(counter), neighbour))

    return []
