#!/usr/bin/env python
"""Train the population of diverse self-play coordination policies and
save it to disk (Section 7 deliverable: "trained cooperative policies").

Usage:
    python -m adhoc_teamwork.scripts.train_population \
        --out results/population --n-policies 3 --n-iterations 200
"""

from __future__ import annotations

import argparse
import time

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.population.policy_pool import train_population
from adhoc_teamwork.training.ppo_trainer import PPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/population")
    parser.add_argument("--n-policies", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iteration", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reward-shaping", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = OvercookedGridworld(horizon=args.horizon, reward_shaping=not args.no_reward_shaping)
    cfg = PPOConfig(hidden_size=args.hidden_size, entropy_coef=args.entropy_coef)

    t0 = time.time()
    pool = train_population(
        env,
        n_policies=args.n_policies,
        n_iterations=args.n_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        cfg=cfg,
        base_seed=args.seed,
        verbose=not args.quiet,
    )
    pool.save(args.out)
    print(f"Saved {len(pool)} policies to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
