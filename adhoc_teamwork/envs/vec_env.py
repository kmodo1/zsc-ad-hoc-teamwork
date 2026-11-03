"""Steps several copies of a :class:`TwoAgentEnv` in lock-step.

Every environment in this project has a fixed episode length, so all copies
finish together and no auto-reset logic is needed. Combined with batched
network inference (see ``policies/agent_api.py``), this is what makes
training and evaluating on the real, 400-step Overcooked env affordable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from adhoc_teamwork.envs.base import TwoAgentEnv


def as_env_fn(env: Union[TwoAgentEnv, Callable[[], TwoAgentEnv]]) -> Callable[[], TwoAgentEnv]:
    """Accept either an env constructor or an env instance to copy."""
    return env if callable(env) else (lambda: copy.deepcopy(env))


@dataclass
class VecStep:
    obs: np.ndarray  # (N, 2, obs_dim)
    rewards: np.ndarray  # (N, 2): the training signal
    team_rewards: np.ndarray  # (N,): the reward to report
    dones: np.ndarray  # (N,)
    legal: np.ndarray  # (N, 2, num_actions)


class VecEnv:
    def __init__(self, env_fn: Callable[[], TwoAgentEnv], n: int):
        self.envs: List[TwoAgentEnv] = [env_fn() for _ in range(n)]
        self.n = n
        self.obs_dim = self.envs[0].obs_dim
        self.num_actions = self.envs[0].num_actions

    def _legal(self) -> np.ndarray:
        return np.stack([e.legal_actions() for e in self.envs])

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack([e.reset(seed=None if seed is None else seed + i) for i, e in enumerate(self.envs)])
        return obs, self._legal()

    def step(self, actions: np.ndarray) -> VecStep:
        results = [e.step([int(a) for a in actions[i]]) for i, e in enumerate(self.envs)]
        return VecStep(
            obs=np.stack([r.obs for r in results]),
            rewards=np.stack([r.rewards for r in results]),
            team_rewards=np.array([r.rewards[0] if r.team_reward is None else r.team_reward for r in results], dtype=np.float32),
            dones=np.array([r.done for r in results]),
            legal=self._legal(),
        )

    @property
    def num_delivered(self) -> np.ndarray:
        return np.array([e.num_delivered for e in self.envs])
