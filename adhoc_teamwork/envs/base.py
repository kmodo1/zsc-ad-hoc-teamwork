"""Common two-agent cooperative environment interface.

Every environment used in this project (the lightweight gridworld in
``gridworld_overcooked.py`` as well as the real Overcooked env from
``ZSC-Eval`` via ``zsceval_adapter.py``) implements this small, explicit
interface so that policies, the trajectory encoder, and the evaluation code
never need to know which concrete environment they are talking to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

NUM_AGENTS = 2


@dataclass
class StepResult:
    """Result of stepping both agents forward by one timestep."""

    obs: np.ndarray  # (NUM_AGENTS, obs_dim)
    rewards: np.ndarray  # (NUM_AGENTS,) -- fully cooperative: identical shared reward
    done: bool
    info: Dict


class TwoAgentEnv:
    """Abstract base class for a fully-cooperative two-agent environment.

    Subclasses must define ``obs_dim`` and ``num_actions`` and implement
    ``reset`` and ``step``.
    """

    obs_dim: int
    num_actions: int
    num_agents: int = NUM_AGENTS

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Return initial observations, shape (num_agents, obs_dim)."""
        raise NotImplementedError

    def step(self, actions: List[int]) -> StepResult:
        """Advance the environment given one action per agent."""
        raise NotImplementedError

    def legal_actions(self) -> np.ndarray:
        """Optional: (num_agents, num_actions) boolean mask. Default: all legal."""
        return np.ones((self.num_agents, self.num_actions), dtype=bool)
