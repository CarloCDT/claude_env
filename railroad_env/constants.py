"""Game constants, ported 1:1 from the original Java referee.

Every value here mirrors a constant in `SummerChallenge2026/src/main/java/com/codingame/game/
Game.java` (or Tile.java). Names are kept close to the originals so the two can be diffed by eye.
"""

# --- Grid dimensions (Game.java) ---
MIN_GRID_HEIGHT = 14
MAX_GRID_HEIGHT = 20
ASPECT_RATIO = 1.5
# The generator only ever produces round(h * 1.5) for h in [14, 20], i.e. these exact pairs:
#   14->21, 15->23, 16->24, 17->26, 18->27, 19->29, 20->30
MAX_GRID_WIDTH = 30

# --- Map generation (Game.java) ---
MIN_TOWN_DISTANCE = 4
AVERAGE_TILES_PER_ZONE_COEFF_TO_GRID_HEIGHT = 2
AVERAGE_TILES_PER_TOWN = 50
RIVER_SPLIT_PROBA = 0.055
RIVER_TO_LAND_MIN_RATIO = 0.07
MIN_MOUNTAINS = 2
MOUNTAIN_TO_CELL_RATIO = 0.04
MIN_RIVER_LENGTH = 3

# --- Economy (Game.java) ---
BASE_RAIL_COST = 1
PASSIVE_INCOME = 3            # paint points granted at the start of every turn
GRASS_COST_MULTIPLIER = 1
RIVER_COST_MULTIPLIER = 2
MOUNTAIN_COST_MULTIPLIER = 3
POI_COST_MULTIPLIER = 3
BLOT_POINTS_PER_TURN = 1      # disruption points granted at the start of every turn
MAX_TURNS = 100

INSTABILITY_THRESHOLD_BASE = 4
INSTABILITY_THRESHOLD_INCREASE = 0

# --- Tile types / track ownership (Tile.java) ---
TYPE_GRASS = 0
TYPE_WATER = 1
TYPE_MOUNTAIN = 2
TYPE_POI = 3

TRACK_NONE = -1
TRACK_NEUTRAL = 2
TOWN_NONE = -1

# --- Directions (Direction.java) ---
# Order matters: it is the tie-break priority for connection pathfinding (N, E, S, W) and the
# iteration order of Grid.ADJACENCY.
NORTH = (0, -1)
EAST = (1, 0)
SOUTH = (0, 1)
WEST = (-1, 0)
ADJACENCY = (NORTH, EAST, SOUTH, WEST)
ADJACENCY_8 = ADJACENCY + ((-1, -1), (1, 1), (1, -1), (-1, 1))
