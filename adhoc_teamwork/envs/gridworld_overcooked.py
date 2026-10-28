"""A small, dependency-light, Overcooked-style cooperative gridworld.

This is *not* a replacement for the real Overcooked environment shipped in
``ZSC-Eval`` -- it exists so that the full research pipeline in this
repository (population training, cross-play evaluation, the partner
trajectory encoder, and the adaptive ensemble) can be implemented, run, and
unit tested on a laptop without a C-extension cooking renderer or a pinned
``gym==0.22`` / ``numpy==1.22`` toolchain.

It keeps the essential structure that makes ZSC interesting in the real
Overcooked environment: the two agent roles are *symmetric* (either agent
could fetch onions or run the pot), so independently self-played pairs can
settle on incompatible conventions about who does what -- exactly the
failure mode described in Carroll et al. (2019) and Hu et al. (2020).

Layout (5 wide x 4 tall; interior floor is the 3x2 block in the middle)::

    X O X D X
    X . . . X     <- O = onion dispenser, D = dish dispenser (on the walls)
    X . . . X     <- P = pot, S = delivery/serve window (on the walls)
    X P X S X

Agents occupy one of the six interior floor cells, face one of four
directions, and may hold at most one item (``None``, ``"onion"``,
``"dish"``, or ``"soup"``). Interacting while facing a dispenser / pot /
delivery tile performs the corresponding Overcooked-style action.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import StepResult, TwoAgentEnv

# Grid dimensions (x in [0, WIDTH), y in [0, HEIGHT))
WIDTH = 5
HEIGHT = 4

# Special wall tiles (not walkable; interact with them from an adjacent cell)
ONION_DISPENSER = (1, 0)
DISH_DISPENSER = (3, 0)
POT = (1, 3)
DELIVERY = (3, 3)
SPECIAL_TILES = {ONION_DISPENSER, DISH_DISPENSER, POT, DELIVERY}

# Interior floor cells agents may stand on
FLOOR_CELLS = [(x, y) for x in (1, 2, 3) for y in (1, 2)]

# Actions
UP, DOWN, LEFT, RIGHT, STAY, INTERACT = range(6)
NUM_ACTIONS = 6
ACTION_NAMES = ["up", "down", "left", "right", "stay", "interact"]
_MOVE_DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
# Orientation index order matches the four movement directions above.
NUM_DIRECTIONS = 4

HELD_ITEMS = [None, "onion", "dish", "soup"]
HELD_INDEX = {item: i for i, item in enumerate(HELD_ITEMS)}

POT_CAPACITY = 3
COOK_TIME = 5
DELIVERY_REWARD = 20.0
STEP_PENALTY = -0.01
# Potential-based shaping terms for subtask completion, used only during
# population training (see `reward_shaping=True`) to make the sparse
# delivery chain (3x onion -> cook -> dish -> serve) tractable for PPO
# within a modest compute budget; evaluation always uses the raw sparse
# reward below so reported returns are directly comparable across methods.
SHAPED_PLACE_ONION_REWARD = 3.0
SHAPED_PICKUP_SOUP_REWARD = 5.0

# Observation layout: self_pos(2) + self_dir(4) + self_held(4)
#                    + other_pos(2) + other_dir(4) + other_held(4)
#                    + pot_onions(1) + pot_cooking(1) + pot_ready(1) + time_left(1)
OBS_DIM = 2 + 4 + 4 + 2 + 4 + 4 + 1 + 1 + 1 + 1


def _is_wall(pos: Tuple[int, int]) -> bool:
    x, y = pos
    if pos in SPECIAL_TILES:
        return True
    if x <= 0 or x >= WIDTH - 1 or y <= 0 or y >= HEIGHT - 1:
        return True
    return False


class OvercookedGridworld(TwoAgentEnv):
    """Two-agent cooperative cooking gridworld (see module docstring)."""

    obs_dim = OBS_DIM
    num_actions = NUM_ACTIONS

    def __init__(self, horizon: int = 200, reward_shaping: bool = False):
        self.horizon = horizon
        self.reward_shaping = reward_shaping
        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.positions = [FLOOR_CELLS[0], FLOOR_CELLS[-1]]
        self.directions = [DOWN, UP]
        self.held = [None, None]
        self.pot_onions = 0
        self.pot_cooking_timer: Optional[int] = None
        self.pot_ready = False
        self.num_delivered = 0
        return self._all_obs()

    def step(self, actions: List[int]) -> StepResult:
        assert len(actions) == self.num_agents
        self.t += 1
        reward = STEP_PENALTY * 2  # shared team reward, split evenly below

        # 1) Resolve movement for both agents simultaneously.
        targets = list(self.positions)
        for i, a in enumerate(actions):
            if a in _MOVE_DELTA:
                dx, dy = _MOVE_DELTA[a]
                nx, ny = self.positions[i][0] + dx, self.positions[i][1] + dy
                self.directions[i] = a
                if not _is_wall((nx, ny)):
                    targets[i] = (nx, ny)
        # Block moves into the cell the other agent currently occupies, and
        # block both agents from moving into the same (shared) target cell.
        if targets[0] == targets[1] and targets[0] != self.positions[0]:
            targets = list(self.positions)
        else:
            for i, other in ((0, 1), (1, 0)):
                if targets[i] == self.positions[other] and targets[i] != self.positions[i]:
                    targets[i] = self.positions[i]
        self.positions = targets

        # 2) Resolve interactions (order doesn't matter for correctness here
        #    since each interaction only touches that agent's held item and
        #    the single shared pot, and both agents interacting with the pot
        #    on the same tick is handled one-at-a-time below).
        for i, a in enumerate(actions):
            if a == INTERACT:
                reward += self._do_interact(i)

        # 3) Advance pot cooking.
        if self.pot_cooking_timer is not None:
            self.pot_cooking_timer -= 1
            if self.pot_cooking_timer <= 0:
                self.pot_cooking_timer = None
                self.pot_ready = True

        done = self.t >= self.horizon
        per_agent_reward = np.array([reward, reward], dtype=np.float32)
        info = {"num_delivered": self.num_delivered}
        return StepResult(obs=self._all_obs(), rewards=per_agent_reward, done=done, info=info)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _facing_tile(self, i: int) -> Tuple[int, int]:
        dx, dy = _MOVE_DELTA[self.directions[i]]
        x, y = self.positions[i]
        return (x + dx, y + dy)

    def _do_interact(self, i: int) -> float:
        tile = self._facing_tile(i)
        reward = 0.0
        if tile == ONION_DISPENSER and self.held[i] is None:
            self.held[i] = "onion"
        elif tile == DISH_DISPENSER and self.held[i] is None:
            self.held[i] = "dish"
        elif tile == POT:
            if self.held[i] == "onion" and self.pot_onions < POT_CAPACITY and self.pot_cooking_timer is None and not self.pot_ready:
                self.pot_onions += 1
                self.held[i] = None
                if self.pot_onions == POT_CAPACITY:
                    self.pot_cooking_timer = COOK_TIME
                if self.reward_shaping:
                    reward += SHAPED_PLACE_ONION_REWARD
            elif self.held[i] == "dish" and self.pot_ready:
                self.held[i] = "soup"
                self.pot_ready = False
                self.pot_onions = 0
                if self.reward_shaping:
                    reward += SHAPED_PICKUP_SOUP_REWARD
        elif tile == DELIVERY and self.held[i] == "soup":
            self.held[i] = None
            self.num_delivered += 1
            reward += DELIVERY_REWARD
        return reward

    def _agent_obs(self, i: int) -> np.ndarray:
        other = 1 - i
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        obs[idx] = self.positions[i][0] / (WIDTH - 1)
        obs[idx + 1] = self.positions[i][1] / (HEIGHT - 1)
        idx += 2
        obs[idx + self.directions[i]] = 1.0
        idx += NUM_DIRECTIONS
        obs[idx + HELD_INDEX[self.held[i]]] = 1.0
        idx += 4
        obs[idx] = self.positions[other][0] / (WIDTH - 1)
        obs[idx + 1] = self.positions[other][1] / (HEIGHT - 1)
        idx += 2
        obs[idx + self.directions[other]] = 1.0
        idx += NUM_DIRECTIONS
        obs[idx + HELD_INDEX[self.held[other]]] = 1.0
        idx += 4
        obs[idx] = self.pot_onions / POT_CAPACITY
        idx += 1
        obs[idx] = (self.pot_cooking_timer / COOK_TIME) if self.pot_cooking_timer else 0.0
        idx += 1
        obs[idx] = 1.0 if self.pot_ready else 0.0
        idx += 1
        obs[idx] = 1.0 - self.t / self.horizon
        idx += 1
        assert idx == OBS_DIM
        return obs

    def _all_obs(self) -> np.ndarray:
        return np.stack([self._agent_obs(0), self._agent_obs(1)], axis=0)

    def render(self) -> str:
        """Simple ASCII rendering, mainly useful for debugging / demos."""
        grid = [["." for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for (x, y) in [(xx, yy) for xx in range(WIDTH) for yy in range(HEIGHT) if _is_wall((xx, yy))]:
            grid[y][x] = "X"
        gx, gy = ONION_DISPENSER
        grid[gy][gx] = "O"
        gx, gy = DISH_DISPENSER
        grid[gy][gx] = "D"
        gx, gy = POT
        grid[gy][gx] = "P"
        gx, gy = DELIVERY
        grid[gy][gx] = "S"
        for i, (x, y) in enumerate(self.positions):
            grid[y][x] = "01"[i]
        return "\n".join("".join(row) for row in grid)
