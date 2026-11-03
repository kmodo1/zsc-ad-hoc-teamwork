"""Scripted, idiosyncratically-biased agents used as "unseen partners" at
evaluation time (Section 4 of the proposal: "we will compare this method
by playing with randomly generated individual biased agents, which are not
made to cooperate well with anything other than their trained partners").

Each of these commits rigidly to one role (or to noise) regardless of what
its partner does, so pairing two agents with the *same* bias (e.g. two
``"onion_runner"``s) produces poor team return even though both are
individually competent -- the same convention-mismatch failure mode that
motivates the whole project.
"""

from __future__ import annotations

import numpy as np

from adhoc_teamwork.envs.gridworld_overcooked import (
    DELIVERY,
    DISH_DISPENSER,
    DOWN,
    INTERACT,
    LEFT,
    NUM_ACTIONS,
    ONION_DISPENSER,
    POT,
    RIGHT,
    UP,
)
from adhoc_teamwork.policies.agent_api import Agent

# (adjacent floor cell, direction that faces the special tile from that cell)
_ADJACENCY = {
    ONION_DISPENSER: ((1, 1), UP),
    DISH_DISPENSER: ((3, 1), UP),
    POT: ((1, 2), DOWN),
    DELIVERY: ((3, 2), DOWN),
}

ROLES = ("onion_runner", "pot_tender", "random_noise")


def _decode_obs(obs: np.ndarray):
    return {
        "pos": (obs[0], obs[1]),
        "direction": int(np.argmax(obs[2:6])),
        "held": int(np.argmax(obs[6:10])),  # 0 none, 1 onion, 2 dish, 3 soup
        "pot_onions": obs[20],
        "pot_cooking": obs[21],
        "pot_ready": bool(obs[22] > 0.5),
    }


_POT_TENDER_IDLE_CELL = (3, 2)  # away from the pot's only adjacent floor
# cell, so an idling pot-tender never blocks the onion-runner's path.
_ONION_RUNNER_IDLE_CELL = (2, 1)  # likewise, away from (1, 2) so an
# onion-runner with nowhere to put its onion never blocks the pot-tender.


def _move_towards(pos_xy, target_cell) -> int:
    """Movement-only step towards ``target_cell``; STAY once there.

    Moves vertically before horizontally: the two roles' "home" columns
    both sit at x=1 while the dish/delivery side sits at x=3, so crossing
    via the shared middle column (x=2) in row y=2 (rather than row y=1,
    where the onion-runner spends most of its time) keeps the two scripted
    roles from blocking each other in this cramped 3x2 interior.
    """
    x, y = pos_xy
    tx, ty = target_cell
    if y != ty:
        return DOWN if ty > y else UP
    if x != tx:
        return RIGHT if tx > x else LEFT
    from adhoc_teamwork.envs.gridworld_overcooked import STAY

    return STAY


def _navigate_and_interact(pos_xy, direction, target_tile) -> int:
    target_cell, target_dir = _ADJACENCY[target_tile]
    # pos is normalized in the obs; the scripted agent needs real grid coords,
    # so callers pass already-denormalized integer coordinates.
    x, y = pos_xy
    tx, ty = target_cell
    if y != ty:
        return DOWN if ty > y else UP
    if x != tx:
        return RIGHT if tx > x else LEFT
    if direction != target_dir:
        return target_dir
    return INTERACT


class ScriptedBiasedAgent(Agent):
    """A rigid, rule-based partner. ``role`` in :data:`ROLES`."""

    def __init__(self, role: str = "onion_runner", noise: float = 0.05, grid_width: int = 5, grid_height: int = 4):
        assert role in ROLES
        self.role = role
        self.noise = noise
        self._w = grid_width
        self._h = grid_height

    def reset(self, batch_size: int = 1) -> None:
        pass

    def _scripted_action(self, obs: np.ndarray) -> int:
        s = _decode_obs(obs)
        x = int(round(s["pos"][0] * (self._w - 1)))
        y = int(round(s["pos"][1] * (self._h - 1)))

        if self.role == "random_noise":
            return int(np.random.randint(NUM_ACTIONS))

        if self.role == "onion_runner":
            pot_raw_onions = int(round(s["pot_onions"] * 3))
            pot_accepting = (not s["pot_ready"]) and s["pot_cooking"] == 0 and pot_raw_onions < 3
            if s["held"] == 1:  # holding an onion
                if pot_accepting:
                    return _navigate_and_interact((x, y), s["direction"], POT)
                return _move_towards((x, y), _ONION_RUNNER_IDLE_CELL)  # pot full/cooking: wait, don't block it
            if s["held"] == 0:  # empty-handed
                if pot_accepting:
                    return _navigate_and_interact((x, y), s["direction"], ONION_DISPENSER)
                return _move_towards((x, y), _ONION_RUNNER_IDLE_CELL)
            return _move_towards((x, y), _ONION_RUNNER_IDLE_CELL)  # holding dish/soup: not this role's job

        if self.role == "pot_tender":
            if s["held"] == 3:  # soup
                return _navigate_and_interact((x, y), s["direction"], DELIVERY)
            if s["held"] == 2:  # dish
                if s["pot_ready"]:
                    return _navigate_and_interact((x, y), s["direction"], POT)
                return _move_towards((x, y), _POT_TENDER_IDLE_CELL)
            # held == 0 (none)
            if s["pot_ready"]:
                return _navigate_and_interact((x, y), s["direction"], DISH_DISPENSER)
            return _move_towards((x, y), _POT_TENDER_IDLE_CELL)

        raise RuntimeError(f"unknown role {self.role}")

    def act(self, obs: np.ndarray, deterministic: bool = False, available_actions=None) -> int:
        if not deterministic and np.random.rand() < self.noise:
            return int(np.random.randint(NUM_ACTIONS))
        return self._scripted_action(obs)

    def action_probs(self, obs: np.ndarray, available_actions=None) -> np.ndarray:
        probs = np.full(NUM_ACTIONS, self.noise / NUM_ACTIONS, dtype=np.float32)
        probs[self._scripted_action(obs)] += 1.0 - self.noise
        return probs
