#!/usr/bin/env python
"""Validate ``envs/zsceval_adapter.py`` and ``policies/zsceval_policy.py``
against ground truth we did not produce: replay the pairings in ZSC-Eval's
published benchmark results through *our* adapter, wrappers and rollout code,
then compare returns pair by pair and player position by position.

A bug in observation formatting, action ordering, action masking or reward
reporting would show up here as a large disagreement.

Usage (via Slurm, see slurm/run.sbatch):
    python -m adhoc_teamwork.scripts.validate_overcooked --layouts unident_s --out <dir>
    python -m adhoc_teamwork.scripts.validate_overcooked --layouts unident_s random0_medium --prepare-only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from adhoc_teamwork.envs.vec_env import VecEnv
from adhoc_teamwork.envs.zsceval_adapter import make_overcooked_env
from adhoc_teamwork.evaluation.rollout import run_episodes_batched
from adhoc_teamwork.policies.nn_agent import NNAgent
from adhoc_teamwork.policies.zsceval_policy import load_zsceval_policy
from adhoc_teamwork.population.zsceval_pool import benchmark_partner_paths, default_pool_root

DEFAULT_AGENTS = ["sp-1", "sp-2", "traj-S2-s36-1", "hsp-S2-s36-1", "fcp-S2-s36-1", "mep-S2-s36-1"]


def agent_checkpoint(layout: str, agent: str) -> str:
    """Pool path of an agent named as in ZSC-Eval's results files."""
    if agent.startswith("sp-"):
        return f"{layout}/fcp/s1/sp/sp{agent.split('-')[1]}_final_actor.pt"
    if agent.startswith("e3t-"):
        return f"{layout}/e3t/s1/e3t/{agent.split('-')[1]}.pt"
    algo, stage, size, seed = agent.split("-")
    return f"{layout}/{algo}/{stage.lower()}/{algo}-{stage}-{size}/{seed}.pt"


def published_results_path(repo_root: str, layout: str, agent: str) -> str:
    algo = agent.split("-")[0]
    return os.path.join(repo_root, "ZSC-Eval/zsceval/scripts/overcooked/eval/results", layout, algo, f"eval-{agent}.json")


def summarize(rows):
    ours = np.array([r["ours"] for r in rows])
    published = np.array([r["published"] for r in rows])
    return {
        "n_cells": len(rows),
        "ours_mean": float(ours.mean()),
        "published_mean": float(published.mean()),
        "mean_abs_diff": float(np.abs(ours - published).mean()),
        "pearson_r": float(np.corrcoef(ours, published)[0, 1]),
    }


def validate_layout(layout, agents, pool_root, repo_root, device, n_episodes, seed):
    vec_env = VecEnv(lambda: make_overcooked_env(layout), n_episodes)
    partner_paths = benchmark_partner_paths(layout)
    partners = {name: load_zsceval_policy(pool_root, layout, path, device=device, name=name) for name, path in partner_paths.items()}

    rows = []
    env_steps, t0 = 0, time.time()
    for agent_name in agents:
        with open(published_results_path(repo_root, layout, agent_name)) as f:
            published = json.load(f)
        policy = load_zsceval_policy(pool_root, layout, agent_checkpoint(layout, agent_name), device=device, name=agent_name)
        for partner_name, partner_policy in partners.items():
            for position, key in ((0, f"{agent_name}-{partner_name}"), (1, f"{partner_name}-{agent_name}")):
                pair = [NNAgent(policy, device), NNAgent(partner_policy, device)]
                if position == 1:
                    pair.reverse()
                result = run_episodes_batched(vec_env, pair, seed=seed + len(rows))
                env_steps += vec_env.n * vec_env.envs[0].horizon
                rows.append(
                    {
                        "agent": agent_name,
                        "partner": partner_name,
                        "agent_position": position,
                        "ours": float(result.returns.mean()),
                        "ours_stderr": float(result.returns.std() / np.sqrt(len(result.returns))),
                        "published": float(published[f"{key}-eval_ep_sparse_r"]),
                    }
                )
        per_agent = [r for r in rows if r["agent"] == agent_name]
        print(f"[{layout}] {agent_name}: {summarize(per_agent)}  ({env_steps / (time.time() - t0):.0f} env-steps/s)", flush=True)
    return {"overall": summarize(rows), "per_agent": {a: summarize([r for r in rows if r["agent"] == a]) for a in agents}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", default=["unident_s"])
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--pool-root", default=default_pool_root())
    parser.add_argument("--n-episodes", type=int, default=20, help="episodes per (agent, partner, position)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--prepare-only", action="store_true", help="just build each layout's cached motion planner")
    args = parser.parse_args()

    import zsceval

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(zsceval.__file__)))
    repo_root = os.path.dirname(repo_root)
    print(f"zsceval imported from {zsceval.__file__}; device={args.device}; pool={args.pool_root}", flush=True)

    for layout in args.layouts:
        t0 = time.time()
        env = make_overcooked_env(layout)
        print(f"[{layout}] env ready in {time.time() - t0:.1f}s; obs_shape={env.obs_shape} num_actions={env.num_actions} horizon={env.horizon}", flush=True)
    if args.prepare_only:
        return

    device = torch.device(args.device)
    results = {}
    for layout in args.layouts:
        results[layout] = validate_layout(layout, args.agents, args.pool_root, repo_root, device, args.n_episodes, args.seed)
        print(f"[{layout}] OVERALL {results[layout]['overall']}", flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "validation.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out}/validation.json")


if __name__ == "__main__":
    main()
