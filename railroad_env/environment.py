"""Gymnasium wrapper around the faithful rules engine.

The agent is always player 0; player 1 is driven by an `opponent_strategy` (see opponent.py).
Actions use the game's own vocabulary:

    ("WAIT",)                       ("PLACE", x, y)
    ("DISRUPT", region_id)          ("DISRUPT_AT", x, y)
    ("AUTOPLACE", x1, y1, x2, y2)

`env.step({"actions": [...]})` submits a whole turn's worth of them, exactly as a real bot's
semicolon-separated output line would.
"""
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import constants as C
from .game_state import GameState
from .grid_maker import GridMaker, JavaRandom
from .opponent import make_opponent


class RailroadGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        grid_height: Optional[int] = None,
        max_turns: int = C.MAX_TURNS,
        opponent_strategy: str = "passive",
        opponent_seed: Optional[int] = None,
        opponent_kwargs: Optional[Dict[str, Any]] = None,
        reward_mode: str = "own",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        cell_size: int = 20,
    ):
        """`grid_height` pins the map height (width follows the 1.5 aspect ratio, as in the real
        game); leave it None for the authentic random 14-20."""
        self.grid_height = grid_height
        self.max_turns = max_turns
        self.opponent_strategy_name = opponent_strategy
        self.opponent_seed = opponent_seed
        # Passed through to the strategy that accepts them - e.g. wait_probability /
        # disrupt_probability for the "random" opponent.
        self.opponent_kwargs = dict(opponent_kwargs or {})
        # "own" = points gained this turn; "margin" = own gain minus the opponent's.
        if reward_mode not in ("own", "margin"):
            raise ValueError(f"reward_mode must be 'own' or 'margin', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.cell_size = cell_size

        self.rng = np.random.RandomState(seed)
        self.game_state: Optional[GameState] = None
        self.opponent = None
        self.renderer = None

        # A turn is a variable-length list of actions; bounds cover the largest possible map.
        self.action_space = spaces.Sequence(
            spaces.Tuple(
                (
                    spaces.Discrete(4),
                    spaces.Box(0, C.MAX_GRID_WIDTH, shape=(4,), dtype=np.int32),
                )
            )
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(C.MAX_GRID_HEIGHT, C.MAX_GRID_WIDTH, GameState.NUM_CHANNELS),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ gym API

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        maker_random = JavaRandom(int(self.rng.randint(0, 2**31 - 1)))
        grid = GridMaker(maker_random, height=self.grid_height).make()
        self.game_state = GameState(grid, max_turns=self.max_turns)

        opponent_seed = (
            self.opponent_seed
            if self.opponent_seed is not None
            else int(self.rng.randint(0, 2**31 - 1))
        )
        self.opponent = make_opponent(
            self.opponent_strategy_name, seed=opponent_seed, **self.opponent_kwargs
        )

        if self.render_mode == "human" and self.renderer is None:
            from .renderer import GameRenderer  # deferred: only rendering needs pygame
            self.renderer = GameRenderer(self.game_state, cell_size=self.cell_size, headless=False)
        elif self.renderer is not None:
            self.renderer.game_state = self.game_state

        return self.game_state.get_observation(), self._get_info()

    def step(self, action: Dict[str, Any]):
        player_actions = _normalize_actions(action.get("actions", []))
        opponent_actions = self.opponent.get_actions(self.game_state, 1)

        summary = self.game_state.resolve_turn([player_actions, opponent_actions])
        self.game_state.do_income()

        obs = self.game_state.get_observation()
        delta = summary["score_delta"]
        reward = float(delta[0] - delta[1]) if self.reward_mode == "margin" else float(delta[0])
        terminated = self.game_state.is_done()

        info = self._get_info()
        info["player_actions"] = {
            "placed": summary["placed"][0],
            "disrupted": summary["disrupted"][0],
        }
        info["opponent_actions"] = {
            "placed": summary["placed"][1],
            "disrupted": summary["disrupted"][1],
        }
        info["inked"] = summary["inked"]
        info["window_open"] = self.render() if self.render_mode == "human" else True

        return obs, reward, terminated, False, info

    def _get_info(self) -> Dict:
        gs = self.game_state
        return {
            "scores": list(gs.scores),
            "turn": gs.turn,
            "paint_points": list(gs.paint_points),
            "disruption_points": list(gs.disruption_points),
            "towns": gs.towns,
            "width": gs.width,
            "height": gs.height,
        }

    def render(self) -> bool:
        """Returns False once the render window has been closed."""
        if self.render_mode == "human" and self.renderer is not None:
            return self.renderer.render()
        return True

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------ helpers

    def legal_placements(self) -> List[Tuple[int, int]]:
        gs = self.game_state
        budget = gs.paint_points[0]
        return [
            (x, y)
            for y in range(gs.height)
            for x in range(gs.width)
            if gs._is_placeable(x, y) and gs.get_track_cost(x, y) <= budget
        ]

    def legal_disruptions(self) -> List[int]:
        gs = self.game_state
        if gs.disruption_points[0] <= 0:
            return []
        return [z.id for z in gs.zones if gs.can_disrupt(z.id)]


def _normalize_actions(actions) -> List[tuple]:
    """Accepts the string-tuple form directly, plus a numeric `(type, params)` form for
    convenience: 0=WAIT, 1=PLACE (x, y), 2=DISRUPT (region_id, _), 3=AUTOPLACE (x1,y1,x2,y2)."""
    normalized = []
    for action in actions:
        if not action:
            continue
        head = action[0]
        if isinstance(head, str):
            normalized.append(tuple(action))
            continue

        params = action[1] if len(action) > 1 else ()
        if head == 0:
            normalized.append(("WAIT",))
        elif head == 1:
            normalized.append(("PLACE", int(params[0]), int(params[1])))
        elif head == 2:
            normalized.append(("DISRUPT", int(params[0])))
        elif head == 3:
            normalized.append(
                ("AUTOPLACE", int(params[0]), int(params[1]), int(params[2]), int(params[3]))
            )
    return normalized
