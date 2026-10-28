"""Builds the NxN cross-play compatibility matrix described in Section 4 of
the proposal: "we will evaluate each policy with different training
partners to create a cross-play compatibility matrix. This will show which
policies work best with which types of partners".
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from adhoc_teamwork.envs.base import TwoAgentEnv
from adhoc_teamwork.evaluation.rollout import evaluate_pair
from adhoc_teamwork.policies.nn_agent import NNAgent
from adhoc_teamwork.population.policy_pool import PolicyPool


def build_crossplay_matrix(env: TwoAgentEnv, pool: PolicyPool, n_episodes: int = 5, seed0: int = 1000) -> Dict:
    """Returns {"names": [...], "matrix": NxN mean-return array (row plays
    agent-0 slot, column plays agent-1 slot)}.
    """
    names: List[str] = pool.names()
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float32)
    delivered = np.zeros((n, n), dtype=np.float32)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            agent_i = NNAgent(pool[name_i])
            agent_j = NNAgent(pool[name_j])
            result = evaluate_pair(env, [agent_i, agent_j], n_episodes=n_episodes, seed0=seed0)
            matrix[i, j] = result["mean_return"]
            delivered[i, j] = result["mean_delivered"]
    return {"names": names, "matrix": matrix, "delivered": delivered}


def self_play_vs_cross_play_gap(crossplay_result: Dict) -> Dict:
    """Quantify the self-play / cross-play gap documented by Carroll et al.
    (2019): the diagonal (self-play, i.e. each policy paired with itself)
    vs. the off-diagonal mean (cross-play, paired with a *different*
    independently-trained policy).
    """
    matrix = crossplay_result["matrix"]
    n = matrix.shape[0]
    diag = np.diag(matrix)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[off_diag_mask] if n > 1 else np.array([])
    return {
        "self_play_mean": float(diag.mean()),
        "cross_play_mean": float(off_diag.mean()) if len(off_diag) else float("nan"),
        "gap": float(diag.mean() - off_diag.mean()) if len(off_diag) else float("nan"),
    }
