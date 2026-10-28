"""Generic episode rollout used everywhere two :class:`Agent` instances need
to be played against each other at inference time: the cross-play matrix,
the baselines, adaptive-ensemble training/evaluation, and the ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from adhoc_teamwork.envs.base import TwoAgentEnv
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
        actions = [agents[i].act(obs[i]) for i in range(2)]
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
        traj.team_return += float(result.rewards[0])
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
