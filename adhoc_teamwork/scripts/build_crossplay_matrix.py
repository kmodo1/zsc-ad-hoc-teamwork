#!/usr/bin/env python
"""Build and save the NxN cross-play compatibility matrix for a trained
population (Section 7 deliverable: "a cross-play evaluation matrix").

Usage:
    python -m adhoc_teamwork.scripts.build_crossplay_matrix \
        --population results/population --out results/crossplay
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from adhoc_teamwork.crossplay.evaluate_crossplay import build_crossplay_matrix, self_play_vs_cross_play_gap
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.evaluation.plotting import plot_crossplay_matrix
from adhoc_teamwork.population.policy_pool import PolicyPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["gridworld", "overcooked"], default="gridworld")
    parser.add_argument("--layout", type=str, default="unident_s", help="Overcooked layout (--env overcooked)")
    parser.add_argument("--pool-root", type=str, default=None, help="ZSC-Eval policy pool (--env overcooked)")
    parser.add_argument("--population", type=str, default="results/population", help="directory (gridworld) or population name (overcooked)")
    parser.add_argument("--out", type=str, default="results/crossplay")
    parser.add_argument("--horizon", type=int, default=None, help="default: 150 on gridworld, 400 on overcooked")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    if args.env == "overcooked":
        from adhoc_teamwork import experiment
        from adhoc_teamwork.envs.zsceval_adapter import make_overcooked_env
        from adhoc_teamwork.population.zsceval_pool import load_pool

        pool = load_pool(experiment.population_paths(args.population, args.layout, args.pool_root), args.layout, args.pool_root, device)
        horizon = args.horizon or 400
        env = lambda: make_overcooked_env(args.layout, horizon=horizon)  # noqa: E731
    else:
        pool = PolicyPool.load(args.population)
        env = OvercookedGridworld(horizon=args.horizon or 150)
    result = build_crossplay_matrix(env, pool, n_episodes=args.n_episodes, device=device)
    gap = self_play_vs_cross_play_gap(result)

    np.savez(os.path.join(args.out, "matrix.npz"), matrix=result["matrix"], delivered=result["delivered"], names=result["names"])
    plot_crossplay_matrix(result["names"], result["matrix"], os.path.join(args.out, "crossplay_matrix.png"))
    with open(os.path.join(args.out, "self_play_vs_cross_play_gap.json"), "w") as f:
        json.dump(gap, f, indent=2)

    print("Cross-play matrix (mean team return):")
    print(result["matrix"])
    print("Self-play vs cross-play gap:", gap)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
