"""Fixed-shape network input encoding.

The real game randomizes the board every match (height 14-20, width = round(height * 1.5), so
21x14 up to 30x20), but the network needs one fixed input shape. Every board is centered in a
BOARD_HEIGHT x BOARD_WIDTH canvas and the off-board padding is marked as inked - i.e. dead
space the agent can never build on - so a padded cell is indistinguishable from a destroyed
region as far as legality is concerned.
"""
from typing import Iterable, Optional, Set, Tuple

import numpy as np

from railroad_env import constants as C
from railroad_env.game_state import GameState

BOARD_HEIGHT = C.MAX_GRID_HEIGHT  # 20
BOARD_WIDTH = C.MAX_GRID_WIDTH    # 30


def compute_pad_offsets(height: int, width: int) -> Tuple[int, int]:
    """Top/left padding that centers a (height, width) board in the fixed canvas."""
    return (BOARD_HEIGHT - height) // 2, (BOARD_WIDTH - width) // 2


def encode_state(game_state: GameState) -> np.ndarray:
    """(NUM_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH) float32, channels-first."""
    obs = game_state.get_observation()  # (h, w, C)
    h, w, _ = obs.shape
    pad_top, pad_left = compute_pad_offsets(h, w)

    canvas = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, obs.shape[2]), dtype=np.float32)
    canvas[pad_top:pad_top + h, pad_left:pad_left + w, :] = obs

    # Mark everything outside the real board as inked, so padding reads as permanently
    # unbuildable rather than as empty plains.
    on_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    on_board[pad_top:pad_top + h, pad_left:pad_left + w] = True
    canvas[:, :, GameState.INKED_CHANNEL] = np.where(
        on_board, canvas[:, :, GameState.INKED_CHANNEL], 1.0
    )

    return np.transpose(canvas, (2, 0, 1)).astype(np.float32)


def build_place_mask(
    game_state: GameState,
    paint_budget: float,
    pending: Optional[Iterable[Tuple[int, int]]] = None,
) -> np.ndarray:
    """(1, BOARD_HEIGHT, BOARD_WIDTH) float32 mask, 1.0 = a track may legally be placed there
    with `paint_budget` left. `pending` are cells already claimed earlier in this same turn."""
    pending_set: Set[Tuple[int, int]] = set(pending or ())
    pad_top, pad_left = compute_pad_offsets(game_state.height, game_state.width)
    mask = np.zeros((1, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

    if paint_budget <= 0:
        return mask

    for y in range(game_state.height):
        for x in range(game_state.width):
            if (x, y) in pending_set:
                continue
            if not game_state._is_placeable(x, y):
                continue
            if game_state.get_track_cost(x, y) > paint_budget:
                continue
            mask[0, pad_top + y, pad_left + x] = 1.0

    return mask


def build_disrupt_mask(game_state: GameState, disruption_budget: int) -> np.ndarray:
    """(1, BOARD_HEIGHT, BOARD_WIDTH) float32 mask, 1.0 = this cell's region may be disrupted.
    Disruption targets a region, so every cell of a legal region is marked."""
    pad_top, pad_left = compute_pad_offsets(game_state.height, game_state.width)
    mask = np.zeros((1, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

    if disruption_budget <= 0:
        return mask

    for zone in game_state.zones:
        if not game_state.can_disrupt(zone.id):
            continue
        for (x, y) in zone.coords:
            mask[0, pad_top + y, pad_left + x] = 1.0

    return mask


def board_coord_from_index(flat_index: int, game_state: GameState) -> Tuple[int, int]:
    """Inverse of the canvas layout: flat index over (BOARD_HEIGHT, BOARD_WIDTH) -> board (x, y)."""
    pad_top, pad_left = compute_pad_offsets(game_state.height, game_state.width)
    row, col = divmod(int(flat_index), BOARD_WIDTH)
    return col - pad_left, row - pad_top
