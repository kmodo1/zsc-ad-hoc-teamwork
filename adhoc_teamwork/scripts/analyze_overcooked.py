#!/usr/bin/env python
"""Summarize a real-Overcooked experiment directory into tables and plots.

Expects the layout produced by ``slurm/submit_overcooked.sh``::

    <root>/<layout>/baselines/results.jsonl
    <root>/<layout>/<population>/population_eval/results.jsonl
    <root>/<layout>/<population>/<arm>/seed<S>/eval/results.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

from adhoc_teamwork.evaluation.aggregate import (
    BASELINE_FAMILIES,
    best_fixed_sub_policy,
    family_partner_scores,
    group_summary,
    mean_over_members,
    oracle_best_sub_policy,
    paired_bootstrap_difference,
    population_steps,
    uniform_vs_baselines,
)
from adhoc_teamwork.evaluation.benchmark import _load_done, method_partner_means
from adhoc_teamwork.evaluation.plotting import plot_family_comparison, plot_weight_trace_grid


POPULATION_FAMILIES = ("single_sub_policy", "single_sub_policy_argmax", "static_ensemble", "static_ensemble_argmax", "static_ensemble_vote")


def load_table(path: str) -> Dict[str, Dict[str, float]]:
    return method_partner_means(_load_done(path).values())


def complete_partners(partners: List[str], scores: Dict[str, List[float]]) -> bool:
    """True once every member has a result for every partner: an experiment
    can be analyzed while it is still running, leaving out unfinished methods.
    """
    return all(p in scores for p in partners) and len({len(scores[p]) for p in partners}) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--population", default="traj-s10")
    parser.add_argument("--out", default=None, help="default: <root>/<layout>/<population>/analysis")
    args = parser.parse_args()

    layout_dir = os.path.join(args.root, args.layout)
    population_dir = os.path.join(layout_dir, args.population)
    out = args.out or os.path.join(population_dir, "analysis")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(population_dir, "population_eval", "partner_groups.json")) as f:
        partner_groups: Dict[str, List[str]] = json.load(f)

    population_table = load_table(os.path.join(population_dir, "population_eval", "results.jsonl"))
    tables = [population_table]
    baselines_path = os.path.join(layout_dir, "baselines", "results.jsonl")
    if os.path.exists(baselines_path):
        tables.append(load_table(baselines_path))
    scores = family_partner_scores(tables)
    argmax_baselines_path = os.path.join(layout_dir, "baselines_argmax", "results.jsonl")
    if os.path.exists(argmax_baselines_path):
        for family, family_scores in family_partner_scores([load_table(argmax_baselines_path)]).items():
            scores[f"{family}_argmax"] = family_scores
    scores["oracle_best_sub_policy"] = {p: [v] for p, v in oracle_best_sub_policy(population_table).items()}
    fixed = best_fixed_sub_policy(population_table, partner_groups["biased"])
    scores["best_fixed_sub_policy"] = {p: [v] for p, v in fixed.items()}

    # Each training arm is its own family, with its training seeds as members.
    arms = sorted(d for d in os.listdir(population_dir) if glob.glob(os.path.join(population_dir, d, "seed*", "eval", "results.jsonl")))
    for arm in arms:
        for path in sorted(glob.glob(os.path.join(population_dir, arm, "seed*", "eval", "results.jsonl"))):
            table = load_table(path)
            for method, partners in table.items():
                family = scores.setdefault(f"{arm}/{method}", {})
                for partner, value in partners.items():
                    family.setdefault(partner, []).append(value)
            # The sub-policy training favored most. Unlike best_fixed_sub_policy,
            # this choice uses no evaluation results, so it is a fair comparison.
            weights_path = os.path.join(os.path.dirname(path), "learned_static_weights.json")
            if os.path.exists(weights_path):
                with open(weights_path) as f:
                    weights = json.load(f)
                top = [m for m in population_table if m.startswith("sub_")][int(np.argmax(weights))]
                family = scores.setdefault(f"{arm}/top_learned_sub_policy", {})
                for partner, value in population_table[top].items():
                    family.setdefault(partner, []).append(value)

    summary = {"layout": args.layout, "population": args.population, "groups": {}, "comparisons": {}}
    for group, partners in partner_groups.items():
        rows = {family: group_summary(s, partners) for family, s in scores.items() if complete_partners(partners, s)}
        summary["groups"][group] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["mean"]))

    # Paired comparisons on the primary group: is the adaptive ensemble better
    # than its two ablations and than each baseline, across biased partners?
    primary = partner_groups["biased"]
    reference_families = [
        f
        for f in (
            "static_ensemble", "static_ensemble_argmax", "single_sub_policy", "best_fixed_sub_policy", "oracle_best_sub_policy",
            "mappo", "traj_s36", "hsp_s36", "fcp_s36", "mep_s36", "cole_s75", "e3t",
            "mappo_argmax", "traj_s36_argmax", "hsp_s36_argmax", "fcp_s36_argmax", "mep_s36_argmax", "cole_s75_argmax", "e3t_argmax",
        )  # fmt: skip
        if f in scores
    ]
    for arm in arms:
        learned_static = f"{arm}/learned_static_ensemble"
        for method in ("adaptive_ensemble", "adaptive_ensemble_argmax", "policy_selection", "learned_static_ensemble"):
            family = f"{arm}/{method}"
            if family not in scores or not complete_partners(primary, scores[family]):
                continue
            ours = mean_over_members(scores[family])
            references = [learned_static, f"{arm}/top_learned_sub_policy"] if method != "learned_static_ensemble" and learned_static in scores else []
            references += reference_families
            summary["comparisons"][family] = {
                ref: paired_bootstrap_difference(ours, mean_over_members(scores[ref]), primary) for ref in references if complete_partners(primary, scores[ref])
            }

    # Ablations: the same method and partners, complete encoder minus ablated one.
    summary["ablations"] = {}
    for arm in arms:
        for suffix in ("-memoryless", "-no_partner_action"):
            base = arm[: -len(suffix)]
            if not arm.endswith(suffix) or base not in arms:
                continue
            for method in ("adaptive_ensemble", "adaptive_ensemble_argmax", "policy_selection"):
                complete, ablated = scores.get(f"{base}/{method}"), scores.get(f"{arm}/{method}")
                if complete and ablated and complete_partners(primary, complete) and complete_partners(primary, ablated):
                    summary["ablations"][f"{arm}/{method}"] = paired_bootstrap_difference(mean_over_members(complete), mean_over_members(ablated), primary)

    # Training for the argmax rule: the anneal-rule arm (sampling temperature
    # annealed), minus the same arm trained at temperature 1.
    summary["argmaxtrain"] = {}
    suffix = "-anneal"
    for arm in arms:
        base = arm[: -len(suffix)]
        if not arm.endswith(suffix) or base not in arms:
            continue
        for method in ("adaptive_ensemble", "adaptive_ensemble_argmax", "policy_selection", "learned_static_ensemble"):
            annealed, plain = scores.get(f"{arm}/{method}"), scores.get(f"{base}/{method}")
            if annealed and plain and complete_partners(primary, annealed) and complete_partners(primary, plain):
                summary["argmaxtrain"][f"{arm}/{method}"] = paired_bootstrap_difference(mean_over_members(annealed), mean_over_members(plain), primary)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    biased_rows = summary["groups"]["biased"]
    steps = {
        "means": {f: biased_rows[f]["mean"] for f in POPULATION_FAMILIES if f in biased_rows},
        "baseline_means": {f: row["mean"] for f, row in biased_rows.items() if f.removesuffix("_argmax") in BASELINE_FAMILIES},
        "steps": population_steps(scores, primary),
        "vs_baselines": uniform_vs_baselines(scores, primary),
    }
    with open(os.path.join(out, "population_steps.json"), "w") as f:
        json.dump(steps, f, indent=2)

    for group, rows in summary["groups"].items():
        print(f"\n## {args.layout} / {args.population} / partners: {group} ({len(partner_groups[group])})\n")
        print("| method | mean return | std across seeds | seeds |\n|---|---:|---:|---:|")
        for family, row in rows.items():
            print(f"| {family} | {row['mean']:.1f} | {row['std_across_seeds']:.1f} | {row['n_seeds']} |")
    if summary["comparisons"]:
        print("\n## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)\n")
        print("| ours | reference | difference | 95% CI | partners better |\n|---|---|---:|---|---:|")
        for family, comparisons in summary["comparisons"].items():
            for ref, c in comparisons.items():
                print(f"| {family} | {ref} | {c['mean_difference']:+.1f} | [{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] | {c['partners_better']}/{c['n_partners']} |")

    if summary["ablations"]:
        print("\n## Ablations on biased partners (complete encoder - ablated, 95% bootstrap CI over distinct partner agents)\n")
        print("| ablated run | difference | 95% CI | partners better |\n|---|---:|---|---:|")
        for family, c in summary["ablations"].items():
            print(f"| {family} | {c['mean_difference']:+.1f} | [{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] | {c['partners_better']}/{c['n_partners']} |")

    if summary["argmaxtrain"]:
        print("\n## Annealed-temperature training on biased partners (anneal arm - the same arm trained at temperature 1, 95% bootstrap CI over distinct partner agents)\n")
        print("| anneal run | difference | 95% CI | partners better |\n|---|---:|---|---:|")
        for family, c in summary["argmaxtrain"].items():
            print(f"| {family} | {c['mean_difference']:+.1f} | [{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] | {c['partners_better']}/{c['n_partners']} |")

    plot_family_comparison(summary["groups"]["biased"], os.path.join(out, "method_comparison_biased.png"), title=f"{args.layout}: mean return with held-out biased partners")
    for arm in arms:
        traces = sorted(glob.glob(os.path.join(population_dir, arm, "seed*", "eval", "weight_traces.npz")))
        if traces:
            data = np.load(traces[0])
            names = [m[len("sub_"):] for m in population_table if m.startswith("sub_")]
            plot_weight_trace_grid({k: data[k] for k in partner_groups["biased"] if k in data.files}, names, os.path.join(out, f"weight_traces_{arm.replace('/', '_')}.png"), title=f"{arm}: ensemble weights over an episode")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
