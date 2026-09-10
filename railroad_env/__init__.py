from .constants import MAX_GRID_HEIGHT, MAX_GRID_WIDTH
from .environment import RailroadGymEnv
from .game_state import GameState
from .grid import Grid, Tile, Town, Zone, get_rail_cost
from .grid_maker import GridMaker, JavaRandom
from .opponent import (
    BOSS_TIERS,
    BossOpponent,
    GreedyAutoplaceOpponent,
    GreedyOpponent,
    Level1ProOpponent,
    Level2ProMaxOpponent,
    OpponentStrategy,
    PassiveOpponent,
    RandomOpponent,
    make_opponent,
)


def __getattr__(name):
    """GameRenderer is imported on first use, not at package import.

    It is the only thing here that needs pygame, and it is only needed to *watch* a game. Import
    it eagerly and anyone who just wants to train a model has to install a graphics stack they
    will never call. Module-level __getattr__ (PEP 562) keeps `from railroad_env import
    GameRenderer` working for people who do have pygame."""
    if name == "GameRenderer":
        from .renderer import GameRenderer
        return GameRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RailroadGymEnv",
    "GameState",
    "Grid",
    "Tile",
    "Town",
    "Zone",
    "GridMaker",
    "JavaRandom",
    "get_rail_cost",
    "MAX_GRID_HEIGHT",
    "MAX_GRID_WIDTH",
    "OpponentStrategy",
    "PassiveOpponent",
    "BossOpponent",
    "RandomOpponent",
    "GreedyOpponent",
    "GreedyAutoplaceOpponent",
    "Level1ProOpponent",
    "Level2ProMaxOpponent",
    "BOSS_TIERS",
    "make_opponent",
    "GameRenderer",
]
