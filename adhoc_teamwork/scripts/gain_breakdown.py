#!/usr/bin/env python
"""Split each population's gain on the held-out biased partners into steps:
single member -> uniform ensemble -> argmax actions -> weights from the current
state (no memory) -> memory of the partner's trajectory.

Reads, for each <results>/<layout>/<population>/:
  population_steps.json  ensembling and argmax, from analyze_overcooked
  summary.json           the two weighting steps, from analyze_overcooked

The steps depend on their order, because argmax and weighting overlap, so the
weighting steps are shown for both sampled and argmax actions.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

LAYOUTS = ("unident_s", "random0_medium")
# The training partners the README's main results use for each population.
POPULATION_ARMS = {"traj-full10": "actor_critic-sp", "s2-mix": "actor_critic-sp+biased", "hsp-w1": "actor_critic-sp+biased"}


def weighting_steps(summary: dict, arm: str, method: str) -> Dict[str, Optional[dict]]:
    """Uniform -> memoryless -> complete encoder, for one action rule. Without a
    memoryless run, only the total of the two can be reported."""
    uniform = "static_ensemble_argmax" if method.endswith("_argmax") else "static_ensemble"
    memoryless = f"{arm}-memoryless/{method}"
    all_weights = summary["comparisons"][f"{arm}/{method}"].get(uniform)
    if memoryless not in summary["comparisons"]:
        return {"state_weights": None, "memory": None, "all_weights": all_weights}
    return {
        "state_weights": summary["comparisons"][memoryless].get(uniform),
        "memory": summary["ablations"].get(memoryless),
        "all_weights": all_weights,
    }


def fmt(step: Optional[dict]) -> str:
    if step is None:
        return "not run"
    return f"{step['mean_difference']:+.1f} [{step['ci95_low']:+.1f}, {step['ci95_high']:+.1f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="adhoc_teamwork/results/overcooked")
    args = parser.parse_args()

    rows = []
    for layout in LAYOUTS:
        for population, arm in POPULATION_ARMS.items():
            directory = os.path.join(args.results, layout, population)
            with open(os.path.join(directory, "summary.json")) as f:
                summary = json.load(f)
            with open(os.path.join(directory, "population_steps.json")) as f:
                steps = json.load(f)["steps"]
            rows.append((layout, population, summary, arm, steps))

    print("| Layout | Population | Ensembling, sampled | Argmax on the ensemble | Argmax on one member | Difference |\n|---|---|---:|---:|---:|---:|")
    for layout, population, _, _, steps in rows:
        cells = [fmt(steps.get(k)) for k in ("ensembling_sampled", "argmax_on_ensemble", "argmax_on_single", "argmax_interaction")]
        print(f"| `{layout}` | `{population}` | " + " | ".join(cells) + " |")

    print("\n| Layout | Population | Actions | State-based weights | Memory | Both |\n|---|---|---|---:|---:|---:|")
    for layout, population, summary, arm, _ in rows:
        for label, method in (("Sampled", "adaptive_ensemble"), ("Argmax", "adaptive_ensemble_argmax")):
            w = weighting_steps(summary, arm, method)
            print(f"| `{layout}` | `{population}` | {label} | {fmt(w['state_weights'])} | {fmt(w['memory'])} | {fmt(w['all_weights'])} |")


if __name__ == "__main__":
    main()
