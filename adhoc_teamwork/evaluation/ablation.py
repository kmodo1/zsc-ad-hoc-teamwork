"""Ablation: adaptive (learned) weighting vs. uniform weighting over the
*same* sub-policy ensemble (Section 4: "We may additionally do ablation
tests to test individual components of the algorithm (e.g., with uniform
weighting instead of the adaptive weighting network)").

This isolates the contribution of the weighting network itself, holding
the population of sub-policies fixed -- as opposed to
``baselines/static_ensemble.py`` more generally, this ablation is always
run on the exact same sub-policy list as the adaptive ensemble it's being
compared to.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from adhoc_teamwork.baselines.static_ensemble import StaticEnsembleAgent
from adhoc_teamwork.ensemble.adaptive_ensemble_agent import AdaptiveEnsembleAgent
from adhoc_teamwork.envs.base import TwoAgentEnv
from adhoc_teamwork.evaluation.rollout import run_episode
from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic


def run_ablation(
    env: TwoAgentEnv,
    sub_policies: List[ActorCritic],
    encoder,
    weighting_net,
    partners: List[Agent],
    n_episodes_per_partner: int = 5,
    seed0: int = 9000,
) -> Dict[str, List[float]]:
    results = {"adaptive_weighting": [], "uniform_weighting": []}
    ep = 0
    for partner in partners:
        for e in range(n_episodes_per_partner):
            traj_adaptive = run_episode(env, [AdaptiveEnsembleAgent(sub_policies, encoder, weighting_net), partner], seed=seed0 + ep)
            traj_uniform = run_episode(env, [StaticEnsembleAgent(sub_policies), partner], seed=seed0 + ep)
            results["adaptive_weighting"].append(traj_adaptive.team_return)
            results["uniform_weighting"].append(traj_uniform.team_return)
            ep += 1
    return results


def summarize(results: Dict[str, List[float]]) -> Dict[str, float]:
    return {name: float(np.mean(vals)) for name, vals in results.items()}
