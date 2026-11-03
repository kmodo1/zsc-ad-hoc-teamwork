"""Builds the NxN cross-play compatibility matrix described in Section 4 of
the proposal: "we will evaluate each policy with different training
partners to create a cross-play compatibility matrix. This will show which
policies work best with which types of partners".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from adhoc_teamwork.envs.vec_env import VecEnv, as_env_fn
from adhoc_teamwork.evaluation.rollout import run_episodes_batched
from adhoc_teamwork.policies.nn_agent import NNAgent
from adhoc_teamwork.population.policy_pool import PolicyPool


def build_crossplay_matrix(
    env,
    pool: PolicyPool,
    n_episodes: int = 5,
    seed0: int = 1000,
    column_pool: Optional[PolicyPool] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Returns {"names": row names, "column_names": ..., "matrix": mean-return
    array (row plays agent-0 slot, column plays agent-1 slot), "delivered": ...}.

    ``env`` is a :class:`TwoAgentEnv` or a callable constructing one.
    ``column_pool`` makes the matrix rectangular (e.g. population x partners);
    by default the population plays itself.
    """
    column_pool = column_pool if column_pool is not None else pool
    names: List[str] = pool.names()
    column_names: List[str] = column_pool.names()
    matrix = np.zeros((len(names), len(column_names)), dtype=np.float32)
    delivered = np.zeros_like(matrix)
    vec_env = VecEnv(as_env_fn(env), n_episodes)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(column_names):
            agents = [NNAgent(pool[name_i], device), NNAgent(column_pool[name_j], device)]
            result = run_episodes_batched(vec_env, agents, seed=seed0)
            matrix[i, j] = result.returns.mean()
            delivered[i, j] = result.delivered.mean()
    return {"names": names, "column_names": column_names, "matrix": matrix, "delivered": delivered}


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
