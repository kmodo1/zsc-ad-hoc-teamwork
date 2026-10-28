"""Final evaluation harness (Section 6: "For experimental validation, we
will evaluate on held-out biased partners and multiple random seeds and
compare average cooperative returns across baselines").

Compares, against a fixed set of held-out partners (population members not
used to train the weighting network, plus scripted biased agents):
  - self-play baseline (a single population policy)
  - static (uniform-weight) ensemble
  - policy-selection baseline (hard argmax of the same learned signal)
  - adaptive ensemble (the proposed method)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from adhoc_teamwork.baselines.policy_selection import PolicySelectionAgent
from adhoc_teamwork.baselines.self_play import self_play_baseline_agent
from adhoc_teamwork.baselines.static_ensemble import StaticEnsembleAgent
from adhoc_teamwork.ensemble.adaptive_ensemble_agent import AdaptiveEnsembleAgent
from adhoc_teamwork.envs.base import TwoAgentEnv
from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.evaluation.rollout import run_episode


def evaluate_methods_against_partners(
    env: TwoAgentEnv,
    sub_policies: List[ActorCritic],
    encoder,
    weighting_net,
    held_out_partners: List[Agent],
    self_play_policy: ActorCritic,
    n_episodes_per_partner: int = 5,
    seed0: int = 5000,
) -> Dict[str, List[float]]:
    """Returns {method_name: [team_return for every (partner, episode)]}."""
    methods = {
        "self_play": lambda: self_play_baseline_agent(self_play_policy),
        "static_ensemble": lambda: StaticEnsembleAgent(sub_policies),
        "policy_selection": lambda: PolicySelectionAgent(sub_policies, encoder, weighting_net),
        "adaptive_ensemble": lambda: AdaptiveEnsembleAgent(sub_policies, encoder, weighting_net),
    }
    results: Dict[str, List[float]] = {name: [] for name in methods}
    ep = 0
    for partner in held_out_partners:
        for e in range(n_episodes_per_partner):
            for name, make_agent in methods.items():
                traj = run_episode(env, [make_agent(), partner], seed=seed0 + ep)
                results[name].append(traj.team_return)
            ep += 1
    return results


def summarize(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {
        name: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for name, vals in results.items()
    }
