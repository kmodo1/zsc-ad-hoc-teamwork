"""Generic episode rollout used everywhere two :class:`Agent` instances need
to be played against each other at inference time: the cross-play matrix,
the baselines, adaptive-ensemble training/evaluation, and the ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from adhoc_teamwork.envs.base import TwoAgentEnv
from adhoc_teamwork.envs.vec_env import VecEnv
from adhoc_teamwork.policies.agent_api import Agent


@dataclass
class EpisodeTrajectory:
    team_return: float = 0.0
    num_delivered: int = 0
    length: int = 0
    # per-agent-0's-view history of (own obs, partner's action), used to
    # train/evaluate the partner trajectory encoder.
    partner_obs: List[np.ndarray] = field(default_factory=list)
    partner_actions: List[int] = field(default_factory=list)
    ego_weights: List[np.ndarray] = field(default_factory=list)  # optional, filled by ensemble agents


def run_episode(
    env: TwoAgentEnv,
    agents: List[Agent],
    seed: Optional[int] = None,
    ego_index: int = 0,
    record_partner_trajectory: bool = False,
) -> EpisodeTrajectory:
    """Play one full episode with ``agents[0]`` and ``agents[1]`` controlling
    the two agent slots. ``ego_index`` picks whose perspective is used when
    recording the partner trajectory (for the encoder).
    """
    assert len(agents) == 2
    for a in agents:
        a.reset()
    obs = env.reset(seed=seed)
    traj = EpisodeTrajectory()
    partner_index = 1 - ego_index
    done = False
    while not done:
        legal = env.legal_actions()
        actions = [agents[i].act(obs[i], available_actions=legal[i]) for i in range(2)]
        if record_partner_trajectory:
            traj.partner_obs.append(obs[ego_index].copy())
            traj.partner_actions.append(actions[partner_index])
        if hasattr(agents[ego_index], "last_weights"):
            w = agents[ego_index].last_weights
            if w is not None:
                traj.ego_weights.append(np.asarray(w).copy())
        result = env.step(actions)
        agents[0].observe_partner_action(actions[1])
        agents[1].observe_partner_action(actions[0])
        traj.team_return += float(result.rewards[0] if result.team_reward is None else result.team_reward)
        obs = result.obs
        done = result.done
    traj.num_delivered = env.num_delivered
    traj.length = env.t
    return traj


def evaluate_pair(env: TwoAgentEnv, agents: List[Agent], n_episodes: int = 5, seed0: int = 0) -> dict:
    """Average a pair of agents' team return over several episodes."""
    returns = []
    delivered = []
    for ep in range(n_episodes):
        traj = run_episode(env, agents, seed=seed0 + ep)
        returns.append(traj.team_return)
        delivered.append(traj.num_delivered)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_delivered": float(np.mean(delivered)),
        "returns": returns,
    }


@dataclass
class BatchedEpisodes:
    returns: np.ndarray  # (n,) team return of each parallel episode
    delivered: np.ndarray  # (n,)
    ego_weights: Optional[np.ndarray] = None  # (T, n, K) when agents[ego_index] is an ensemble


def run_episodes_batched(vec_env: VecEnv, agents: List[Agent], seed: Optional[int] = None, ego_index: int = 0, record_weights: bool = False) -> BatchedEpisodes:
    """Play ``vec_env.n`` parallel episodes with ``agents[i]`` in slot ``i``
    of every environment. ``seed`` fixes the agents' action sampling; run
    both orderings of a pair to cover both player positions.
    """
    assert len(agents) == 2
    for a in agents:
        a.reset(batch_size=vec_env.n)
    obs, legal = vec_env.reset(seed=seed)
    if seed is not None:  # after reset: some envs reseed the global RNGs there
        torch.manual_seed(seed)
        np.random.seed(seed)
    returns = np.zeros(vec_env.n, dtype=np.float64)
    weights = []
    while True:
        actions = np.stack([agents[i].act_batch(obs[:, i], available_actions=legal[:, i]) for i in range(2)], axis=1)
        if record_weights and getattr(agents[ego_index], "last_weights", None) is not None:
            weights.append(np.asarray(agents[ego_index].last_weights).reshape(vec_env.n, -1).copy())
        step = vec_env.step(actions)
        agents[0].observe_partner_action(actions[:, 1])
        agents[1].observe_partner_action(actions[:, 0])
        returns += step.team_rewards
        obs, legal = step.obs, step.legal
        if step.dones.all():
            break
    return BatchedEpisodes(returns=returns, delivered=vec_env.num_delivered, ego_weights=np.stack(weights) if weights else None)
