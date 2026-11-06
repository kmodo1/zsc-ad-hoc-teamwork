#!/usr/bin/env python
"""Collect the final checkpoints of policies trained with ``slurm/train_hsp_pair.sh``
and ``slurm/train_traj_population.sh`` into the folder that the ``hsp-w1-new`` and
``traj-new10`` populations read (see ``experiment.py``).

A run that was preempted and requeued leaves an unfinished earlier run beside the
finished one. Only a run that reached the end of training is ever used.

Usage:
    python -m adhoc_teamwork.scripts.extract_new_policies --layout random3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from typing import Dict, Optional, Tuple

from adhoc_teamwork.experiment import new_policy_root
from adhoc_teamwork.population.zsceval_pool import biased_counterpart_paths

TOTAL_STEPS = int(1e7)
# Checkpoints land on rollout boundaries, so the last one can fall just short of the total.
FINISHED_FRACTION = 0.99


def results_dir() -> str:
    return os.environ.get("ZSC_RESULTS_DIR", os.path.expanduser("~/ZSC/results"))


def latest_checkpoint(pattern: str) -> Optional[Tuple[int, str]]:
    """(step, path) of the highest-step checkpoint matching ``pattern``, across runs."""
    best = None
    for path in glob.glob(pattern):
        match = re.search(r"periodic_(\d+)\.pt$", path)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return best


def collect(sources: Dict[str, str], out_dir: str) -> int:
    """Copy each finished checkpoint to ``out_dir/<name>``. Returns how many are ready."""
    os.makedirs(out_dir, exist_ok=True)
    ready = 0
    for name, pattern in sources.items():
        found = latest_checkpoint(pattern)
        if found is None or found[0] < FINISHED_FRACTION * TOTAL_STEPS:
            print(f"  not finished: {name} ({'no checkpoints' if found is None else f'at step {found[0]}'})")
            continue
        shutil.copyfile(found[1], os.path.join(out_dir, name))
        ready += 1
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", required=True)
    args = parser.parse_args()

    runs = os.path.join(results_dir(), "Overcooked", args.layout)
    out = os.path.join(new_policy_root(), args.layout)

    exps = [int(re.match(r"hsp(\d+)_", name).group(1)) for name in biased_counterpart_paths(args.layout, 10)]
    pairs = {}
    for exp in exps:
        for agent, tag in ((0, "w0"), (1, "w1")):  # slot 0 always carries the biased reward
            pairs[f"hsp{exp}_final_{tag}_actor.pt"] = os.path.join(runs, "mappo", f"hsp-new-r1-{exp}", "run*", "models", f"actor_agent{agent}_periodic_*.pt")
    print(f"{args.layout}: {collect(pairs, os.path.join(out, 'hsp'))}/{len(pairs)} HSP checkpoints ready")

    members = {f"traj{m}_final_actor.pt": os.path.join(runs, "traj", "traj-new-S1-s10", "run*", "models", f"traj{m}", "actor_periodic_*.pt") for m in range(1, 11)}
    print(f"{args.layout}: {collect(members, os.path.join(out, 'traj'))}/{len(members)} TrajeDi checkpoints ready")


if __name__ == "__main__":
    main()
