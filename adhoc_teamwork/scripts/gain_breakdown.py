#!/usr/bin/env python
"""Split each population's gain on the held-out biased partners into steps:
single member -> uniform ensemble -> argmax actions -> weights from the current
state (no memory) -> memory of the partner's trajectory.

Reads, for each <results>/<layout>/<population>/:
  population_steps.json  ensembling and argmax, from analyze_overcooked
  summary.json           the two weighting steps, from analyze_overcooked

The steps depend on their order, because argmax and weighting overlap, so the
weighting steps are shown for both sampled and argmax actions.

Layouts without trained ensembles only appear in the tables that need none.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

LAYOUTS = ("unident_s", "random0_medium", "random0", "random1", "random3", "small_corridor")
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
            steps_path = os.path.join(directory, "population_steps.json")
            if not os.path.exists(steps_path):
                continue
            with open(os.path.join(directory, "summary.json")) as f:
                summary = json.load(f)
            with open(steps_path) as f:
                data = json.load(f)
            rows.append((layout, population, summary, arm, data["steps"], data))

    print("| Layout | Population | Ensembling, sampled | Argmax on the ensemble | Argmax on one member | Difference |\n|---|---|---:|---:|---:|---:|")
    for layout, population, _, _, steps, _ in rows:
        cells = [fmt(steps.get(k)) for k in ("ensembling_sampled", "argmax_on_ensemble", "argmax_on_single", "argmax_interaction")]
        print(f"| `{layout}` | `{population}` | " + " | ".join(cells) + " |")

    print("\n| Layout | Population | Actions | State-based weights | Memory | Both |\n|---|---|---|---:|---:|---:|")
    for layout, population, summary, arm, _, _ in rows:
        if f"{arm}/adaptive_ensemble" not in summary["comparisons"]:
            continue
        for label, method in (("Sampled", "adaptive_ensemble"), ("Argmax", "adaptive_ensemble_argmax")):
            w = weighting_steps(summary, arm, method)
            print(f"| `{layout}` | `{population}` | {label} | {fmt(w['state_weights'])} | {fmt(w['memory'])} | {fmt(w['all_weights'])} |")

    print("\n| Layout | Population | Mean argmax | Vote | Vote minus mean argmax |\n|---|---|---:|---:|---:|")
    for layout, population, _, _, steps, data in rows:
        means = data["means"]
        if "static_ensemble_vote" in means:
            print(f"| `{layout}` | `{population}` | {means['static_ensemble_argmax']:.1f} | {means['static_ensemble_vote']:.1f} | {fmt(steps.get('vote_vs_mean_argmax'))} |")

    print("\n| Layout | Population | Uniform, argmax | Best baseline, argmax | Difference | Uniform, sampled | Best baseline, sampled | Difference |\n|---|---|---:|---|---:|---:|---|---:|")
    for layout, population, _, _, _, data in rows:
        cells = []
        for ensemble in ("static_ensemble_argmax", "static_ensemble"):
            comparisons = data.get("vs_baselines", {}).get(ensemble, {})
            if not comparisons:
                cells += ["-", "not run", "-"]
                continue
            best = max(comparisons, key=lambda f: data["baseline_means"][f])
            cells += [f"{data['means'][ensemble]:.1f}", f"{best} ({data['baseline_means'][best]:.1f})", fmt(comparisons[best])]
        print(f"| `{layout}` | `{population}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
