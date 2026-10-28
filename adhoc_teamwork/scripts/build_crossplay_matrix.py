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

from adhoc_teamwork.crossplay.evaluate_crossplay import build_crossplay_matrix, self_play_vs_cross_play_gap
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.evaluation.plotting import plot_crossplay_matrix
from adhoc_teamwork.population.policy_pool import PolicyPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=str, default="results/population")
    parser.add_argument("--out", type=str, default="results/crossplay")
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pool = PolicyPool.load(args.population)
    env = OvercookedGridworld(horizon=args.horizon)
    result = build_crossplay_matrix(env, pool, n_episodes=args.n_episodes)
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
