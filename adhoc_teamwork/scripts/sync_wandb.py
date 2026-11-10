#!/usr/bin/env python
"""Mirror an experiment directory to Weights & Biases, for watching runs live.

This only reads files that the jobs already write (each run's ``train_state.pt``
and evaluation results), so training has no dependency on wandb: it needs no
restart to be picked up, and a network or wandb failure cannot affect it.

Set WANDB_ENTITY and WANDB_PROJECT in the environment (see ``slurm/env.sh``).
"""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import os
import re
import subprocess
import time
from typing import Dict, List

import numpy as np
import torch
import wandb

from adhoc_teamwork.evaluation.aggregate import method_family
from adhoc_teamwork.evaluation.benchmark import _load_done, method_partner_means

JOB_PREFIX = "adhoc-"
SYNC_JOB = "adhoc-wandb-sync"
TRAIN_METRICS = {
    "returns": "train/return",
    "losses": "train/loss",
    "weight_entropy": "weights/entropy",
    "weight_within_episode_std": "weights/std_within_episode",
    "weight_between_episode_std": "weights/std_between_episodes",
    "temperature": "train/temperature",
}


def run_id(relative_path: str) -> str:
    """Stable, so that every sync of a run continues the same wandb run."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", relative_path)[-64:]


def eval_summary(eval_dir: str) -> Dict[str, float]:
    """{"eval/<partner group>/<method>": mean return} for finished methods."""
    results, groups_path = os.path.join(eval_dir, "results.jsonl"), os.path.join(eval_dir, "partner_groups.json")
    if not (os.path.exists(results) and os.path.exists(groups_path)):
        return {}
    with open(groups_path) as f:
        groups: Dict[str, List[str]] = json.load(f)
    table = method_partner_means(_load_done(results).values())
    summary = {}
    for method, partners in table.items():
        for group, names in groups.items():
            if all(n in partners for n in names):
                summary[f"eval/{group}/{method}"] = float(np.mean([partners[n] for n in names]))
    return summary


def family_summary(eval_dir: str) -> Dict[str, float]:
    """Like :func:`eval_summary`, averaging over the seeds of each algorithm."""
    collected: Dict[str, List[float]] = {}
    for key, value in eval_summary(eval_dir).items():
        prefix, group, method = key.split("/", 2)
        collected.setdefault(f"{prefix}/{group}/{method_family(method)}", []).append(value)
    return {k: float(np.mean(v)) for k, v in collected.items()}


def sync_training_run(run_dir: str, root: str, args, state: dict) -> None:
    relative = os.path.relpath(run_dir, root)
    checkpoint = os.path.join(run_dir, "train_state.pt")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False) if os.path.exists(checkpoint) else None
    summary = eval_summary(os.path.join(run_dir, "eval"))
    signature = [ckpt["iteration"] if ckpt else 0, len(summary)]
    if ckpt is None or state.get(relative) == signature:
        return

    layout, population, arm, seed = relative.split(os.sep)
    config_path = os.path.join(run_dir, "config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    run = wandb.init(
        entity=args.entity, project=args.project, id=run_id(relative), resume="allow", name=f"{layout}/{population}/{arm}/{seed}",
        group=f"{layout}/{population}/{arm}", job_type="train", tags=[layout, population, arm], config=config, dir=os.environ.get("WANDB_DIR"),
    )  # fmt: skip
    stats = ckpt["stats"]
    for i in range(run.starting_step, ckpt["iteration"]):
        run.log({name: stats[key][i] for key, name in TRAIN_METRICS.items() if i < len(stats.get(key, []))}, step=i)
    run.summary.update(summary)
    run.finish()
    state[relative] = signature


def sync_reference_run(population_dir: str, root: str, args, state: dict) -> None:
    """Baselines and uniform-weight results, to compare the training runs against."""
    relative = os.path.relpath(population_dir, root)
    layout, population = relative.split(os.sep)
    summary = family_summary(os.path.join(population_dir, "population_eval"))
    summary.update(family_summary(os.path.join(root, layout, "baselines")))
    images = sorted(glob.glob(os.path.join(population_dir, "analysis", "*.png")))
    signature = [len(summary), [int(os.path.getmtime(p)) for p in images]]
    if not summary or state.get(relative) == signature:
        return
    run = wandb.init(
        entity=args.entity, project=args.project, id=run_id(f"{relative}/reference"), resume="allow", name=f"{layout}/{population}/reference",
        group=f"{layout}/{population}/reference", job_type="reference", tags=[layout, population], dir=os.environ.get("WANDB_DIR"),
    )  # fmt: skip
    run.summary.update(summary)
    if images:
        run.log({os.path.splitext(os.path.basename(p))[0]: wandb.Image(p) for p in images})
    run.finish()
    state[relative] = signature


def other_jobs_remain() -> bool:
    output = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], capture_output=True, text=True, check=True).stdout
    return any(j.startswith(JOB_PREFIX) and j != SYNC_JOB for j in output.split())


def sync_all(root: str, args, state: dict) -> None:
    for run_dir in sorted(glob.glob(os.path.join(root, "*", "*", "*", "seed*"))):
        try:
            sync_training_run(run_dir, root, args, state)
        except Exception as e:  # one unreadable run should not stop the others
            print(f"[sync] {run_dir}: {e!r}", flush=True)
    for population_dir in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*", "population_eval"))):
        try:
            sync_reference_run(population_dir, root, args, state)
        except Exception as e:
            print(f"[sync] {population_dir}: {e!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="experiment directory")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--loop", type=int, default=0, help="seconds between syncs, until no other jobs remain (default: sync once)")
    args = parser.parse_args()
    if not args.project:
        parser.error("set WANDB_PROJECT or --project")

    state: dict = {}
    while True:
        finished = args.loop == 0 or not other_jobs_remain()
        sync_all(args.root, args, state)
        print(f"[sync] {time.strftime('%H:%M:%S')}: {len(state)} runs up to date", flush=True)
        if finished:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
