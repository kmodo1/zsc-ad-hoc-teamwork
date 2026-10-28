#!/usr/bin/env python
"""Ablation: learned (adaptive) weighting vs. uniform weighting over the
same sub-policy ensemble.

Usage:
    python -m adhoc_teamwork.scripts.run_ablation \
        --population results/population --ensemble results/ensemble/ensemble.pt \
        --out results/ablation
"""

from __future__ import annotations

import argparse
import json
import os

from adhoc_teamwork.ensemble.checkpoint import load_ensemble_checkpoint
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.evaluation.ablation import run_ablation, summarize
from adhoc_teamwork.evaluation.plotting import plot_method_comparison
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.population.policy_pool import PolicyPool

PARTNER_SPECS = [("onion_runner", 0.3), ("pot_tender", 0.3), ("random_noise", 0.0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=str, default="results/population")
    parser.add_argument("--ensemble", type=str, default="results/ensemble/ensemble.pt")
    parser.add_argument("--out", type=str, default="results/ablation")
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--n-episodes-per-partner", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pool = PolicyPool.load(args.population)
    encoder, weighting_net, held_in = load_ensemble_checkpoint(args.ensemble)
    sub_policies = [pool[name] for name in held_in]

    partners = [ScriptedBiasedAgent(role, noise=noise) for role, noise in PARTNER_SPECS]
    env = OvercookedGridworld(horizon=args.horizon)
    results = run_ablation(env, sub_policies, encoder, weighting_net, partners, n_episodes_per_partner=args.n_episodes_per_partner)
    summary = summarize(results)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_method_comparison(results, os.path.join(args.out, "ablation.png"), title="Adaptive vs. uniform weighting")

    print(json.dumps(summary, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
