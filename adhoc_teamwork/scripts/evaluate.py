#!/usr/bin/env python
"""Final comparison: self-play vs. static ensemble vs. policy-selection vs.
the adaptive ensemble, against held-out partners (Section 6: "evaluate on
held-out biased partners and multiple random seeds"; Section 7 deliverable:
"plots showing performance and policy-weight changes").

Usage:
    python -m adhoc_teamwork.scripts.evaluate \
        --population results/population --ensemble results/ensemble/ensemble.pt \
        --out results/evaluation
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from adhoc_teamwork.ensemble.adaptive_ensemble_agent import AdaptiveEnsembleAgent
from adhoc_teamwork.ensemble.checkpoint import load_ensemble_checkpoint
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.evaluation.plotting import plot_method_comparison, plot_weight_trajectory
from adhoc_teamwork.evaluation.rollout import run_episode
from adhoc_teamwork.evaluation.run_evaluation import evaluate_methods_against_partners, summarize
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.population.policy_pool import PolicyPool

# Held-out evaluation partners: scripted roles/noise levels and configurations
# NOT seen during adaptive-ensemble training (see scripts/train_adaptive_ensemble.py),
# so these numbers measure zero-shot generalization rather than memorization.
HELD_OUT_PARTNER_SPECS = [
    ("onion_runner", 0.3),
    ("pot_tender", 0.3),
    ("random_noise", 0.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=str, default="results/population")
    parser.add_argument("--ensemble", type=str, default="results/ensemble/ensemble.pt")
    parser.add_argument("--out", type=str, default="results/evaluation")
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--n-episodes-per-partner", type=int, default=5)
    parser.add_argument("--held-out-policy", type=str, default=None, help="also evaluate against this held-out population policy")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pool = PolicyPool.load(args.population)
    encoder, weighting_net, held_in = load_ensemble_checkpoint(args.ensemble)
    sub_policies = [pool[name] for name in held_in]

    held_out_partners = [ScriptedBiasedAgent(role, noise=noise) for role, noise in HELD_OUT_PARTNER_SPECS]
    if args.held_out_policy and args.held_out_policy in pool.names():
        from adhoc_teamwork.policies.nn_agent import NNAgent

        held_out_partners.append(NNAgent(pool[args.held_out_policy]))

    env = OvercookedGridworld(horizon=args.horizon)
    results = evaluate_methods_against_partners(
        env,
        sub_policies,
        encoder,
        weighting_net,
        held_out_partners,
        self_play_policy=sub_policies[0],
        n_episodes_per_partner=args.n_episodes_per_partner,
    )
    summary = summarize(results)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_method_comparison(results, os.path.join(args.out, "method_comparison.png"))

    # Also record one sample episode's weight trajectory against a biased partner.
    agent = AdaptiveEnsembleAgent(sub_policies, encoder, weighting_net)
    traj = run_episode(env, [agent, held_out_partners[0]], seed=123)
    if traj.ego_weights:
        plot_weight_trajectory(np.stack(traj.ego_weights), held_in, os.path.join(args.out, "weight_trajectory.png"))

    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
