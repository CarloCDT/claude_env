"""Grid primitives: Tile, Zone, Town, Grid - ported from the Java referee's grid package.

Coordinates are plain `(x, y)` tuples throughout (the Java `Coord`), and the grid is indexed
`grid.get(x, y)`. Numpy arrays elsewhere in this package stay `[y, x]`-indexed as usual.
"""
from typing import Dict, List, Optional, Tuple

from .constants import (
    ADJACENCY,
    TOWN_NONE,
    TRACK_NONE,
    TYPE_GRASS,
    TYPE_MOUNTAIN,
    TYPE_POI,
    TYPE_WATER,
)

Coord = Tuple[int, int]


class Tile:
    __slots__ = ("coord", "type", "zone_id", "track", "town_id", "active_connections")

    def __init__(self, coord: Coord, type_: int = TYPE_GRASS):
        self.coord = coord
        self.type = type_
        self.zone_id = -1
        self.track = TRACK_NONE
        self.town_id = TOWN_NONE
        # List of (from_town_id, to_town_id) pairs this tile is part of - the Java ScheduleStep.
        self.active_connections: List[Tuple[int, int]] = []

    def is_water(self) -> bool:
        return self.type == TYPE_WATER

    def is_mountain(self) -> bool:
        return self.type == TYPE_MOUNTAIN

    def is_plains(self) -> bool:
        return self.type == TYPE_GRASS

    def is_town(self) -> bool:
        return self.town_id != TOWN_NONE

    def is_track(self) -> bool:
        return self.track != TRACK_NONE

    def is_track_or_town(self) -> bool:
        return self.is_town() or self.is_track()


class Zone:
    """A region of contiguous cells. Disruption targets zones, not cells."""

    __slots__ = ("id", "coords", "neighbours", "contained_towns", "instability", "inked")

    def __init__(self, id_: int, coords: List[Coord]):
        self.id = id_
        self.coords: List[Coord] = coords
        self.neighbours: List[int] = []
        self.contained_towns: List["Town"] = []
        self.instability = 0
        self.inked = False


class Town:
    __slots__ = ("id", "coord", "desired_connections", "active_connections", "paths")

    def __init__(self, id_: int, coord: Coord):
        self.id = id_
        self.coord = coord
        # Town objects this town wants connected to itself (unilateral - see GridMaker).
        self.desired_connections: List["Town"] = []
        self.active_connections: List["Town"] = []
        # to_town_id -> list of coords making up the current shortest path
        self.paths: Dict[int, List[Coord]] = {}


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells: Dict[Coord, Tile] = {}
        for y in range(height):
            for x in range(width):
                self.cells[(x, y)] = Tile((x, y))
        self.towns: List[Town] = []
        self.zones: List[Zone] = []
        self.pois: List[Coord] = []

    def get(self, x: int, y: int) -> Optional[Tile]:
        """Returns None for out-of-grid coords (the Java Tile.NO_TILE sentinel)."""
        return self.cells.get((x, y))

    def get_neighbours(self, coord: Coord, adjacency=ADJACENCY) -> List[Coord]:
        x, y = coord
        neighbours = []
        for dx, dy in adjacency:
            n = (x + dx, y + dy)
            if n in self.cells:
                neighbours.append(n)
        return neighbours

    def can_train_pass(self, coord: Coord) -> bool:
        """Whether a connection path may run through this cell: it needs a track (of any owner,
        including neutral) or be a town. Mirrors Grid.canTrainPass - note it deliberately does
        NOT check `inked`, because inking already strips every track in the zone and a zone
        containing a town can never be inked."""
        tile = self.cells.get(coord)
        if tile is None:
            return False
        return tile.is_town() or tile.track != TRACK_NONE

    def clone_terrain(self) -> "Grid":
        new_grid = Grid(self.width, self.height)
        for coord, tile in self.cells.items():
            new_tile = new_grid.cells[coord]
            new_tile.type = tile.type
            new_tile.zone_id = tile.zone_id
        return new_grid


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_rail_cost(tile: Tile) -> int:
    """Game.getRailCost: plains 1, river 2, mountain 3."""
    from .constants import (
        BASE_RAIL_COST,
        GRASS_COST_MULTIPLIER,
        MOUNTAIN_COST_MULTIPLIER,
        POI_COST_MULTIPLIER,
        RIVER_COST_MULTIPLIER,
    )

    if tile.is_mountain():
        return BASE_RAIL_COST * MOUNTAIN_COST_MULTIPLIER
    if tile.is_water():
        return BASE_RAIL_COST * RIVER_COST_MULTIPLIER
    if tile.type == TYPE_POI:
        return BASE_RAIL_COST * POI_COST_MULTIPLIER
    return BASE_RAIL_COST * GRASS_COST_MULTIPLIER
