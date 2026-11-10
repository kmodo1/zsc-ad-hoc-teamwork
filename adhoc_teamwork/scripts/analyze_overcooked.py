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

from adhoc_teamwork.experiment import GROUP_SEEN_BY_BASELINES
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
ENSEMBLE_METHODS = (
    "adaptive_ensemble", "adaptive_ensemble_argmax", "policy_selection", "policy_selection_argmax",
    "learned_static_ensemble", "learned_static_ensemble_argmax",
)  # fmt: skip


def baseline_families_for(group: str, scores: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Baseline families (sampled and argmax) that this group is held out from."""
    seen = set(GROUP_SEEN_BY_BASELINES.get(group, ()))
    families = [f for f in BASELINE_FAMILIES if f not in seen]
    return [f for f in families + [f"{f}_argmax" for f in families] if f in scores]


def best_baseline(group_rows: Dict[str, dict], families: List[str], argmax: bool) -> str:
    """The eligible baseline family with the best mean on a group, for one action rule."""
    candidates = [f for f in families if f.endswith("_argmax") == argmax and f in group_rows]
    return max(candidates, key=lambda f: group_rows[f]["mean"]) if candidates else None


def load_table(path: str) -> Dict[str, Dict[str, float]]:
    return method_partner_means(_load_done(path).values())


def complete_partners(partners: List[str], scores: Dict[str, List[float]]) -> bool:
    """True once every member has a result for every partner: an experiment
    can be analyzed while it is still running, leaving out unfinished methods.
    """
    return all(p in scores for p in partners) and len({len(scores[p]) for p in partners}) == 1


def _rounded(x, digits: int = 3):
    """Keeps summaries small enough to commit."""
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: _rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v, digits) for v in x]
    return x


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
    # The best single member in hindsight, chosen separately for each group.
    fixed_by_group = {group: {p: [v] for p, v in best_fixed_sub_policy(population_table, partners).items()} for group, partners in partner_groups.items()}
    scores["best_fixed_sub_policy"] = fixed_by_group["biased"]

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

    summary = {"layout": args.layout, "population": args.population, "groups": {}, "best_baselines": {}, "group_comparisons": {}}
    for group, partners in partner_groups.items():
        group_scores = {**scores, "best_fixed_sub_policy": fixed_by_group[group]}
        rows = {family: group_summary(s, partners) for family, s in group_scores.items() if complete_partners(partners, s)}
        summary["groups"][group] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["mean"]))

    # Paired comparisons on each group: is the adaptive ensemble better than its
    # ablations and than each baseline the group is held out from? Biased
    # partners are the primary comparison; the others test other generators.
    for group, partners in partner_groups.items():
        group_scores = {**scores, "best_fixed_sub_policy": fixed_by_group[group]}
        families = baseline_families_for(group, scores)
        best = {"best_baseline": best_baseline(summary["groups"][group], families, argmax=False), "best_baseline_argmax": best_baseline(summary["groups"][group], families, argmax=True)}
        summary["best_baselines"][group] = best
        reference_families = [
            f for f in ("static_ensemble", "static_ensemble_argmax", "single_sub_policy", "best_fixed_sub_policy", "oracle_best_sub_policy") if f in group_scores
        ]
        # Every baseline on the primary group; elsewhere only the best, to keep summaries small.
        reference_families += families if group == "biased" else []
        comparisons = summary["group_comparisons"][group] = {}
        for arm in arms:
            for method in ENSEMBLE_METHODS:
                family = f"{arm}/{method}"
                if family not in scores or not complete_partners(partners, scores[family]):
                    continue
                ours = mean_over_members(scores[family])
                references = [] if method.startswith("learned_static") else [f"{arm}/learned_static_ensemble", f"{arm}/learned_static_ensemble_argmax", f"{arm}/top_learned_sub_policy"]
                references = {ref: ref for ref in references + reference_families if ref in group_scores}
                references.update({alias: ref for alias, ref in best.items() if ref is not None})
                comparisons[family] = {
                    alias: paired_bootstrap_difference(ours, mean_over_members(group_scores[ref]), partners)
                    for alias, ref in references.items()
                    if complete_partners(partners, group_scores[ref])
                }
    # The primary group's comparisons keep their original key, and are not stored twice.
    summary["comparisons"] = summary["group_comparisons"].pop("biased")
    primary = partner_groups["biased"]
    # Per-partner means (over seeds) of every family with complete results, so
    # that scripts/report_tables.py can pair any two of them, e.g. two training arms.
    # A partner is kept once every seed of the family has a result for it.
    summary["partner_groups"] = partner_groups
    summary["partner_means"] = {}
    all_partners = [p for g in partner_groups.values() for p in g]
    for family, s in {**scores, **{f"best_fixed_sub_policy@{g}": fixed_by_group[g] for g in partner_groups}}.items():
        have = [p for p in all_partners if p in s]
        if have:
            n = max(len(s[p]) for p in have)
            summary["partner_means"][family] = {p: round(float(np.mean(s[p])), 3) for p in have if len(s[p]) == n}

    # Ablations: the same method and partners, complete encoder minus ablated one.
    summary["ablations"] = {}
    for arm in arms:
        for suffix in ("-memoryless", "-no_partner_action"):
            base = arm[: -len(suffix)]
            if not arm.endswith(suffix) or base not in arms:
                continue
            for method in ("adaptive_ensemble", "adaptive_ensemble_argmax", "policy_selection", "policy_selection_argmax"):
                complete, ablated = scores.get(f"{base}/{method}"), scores.get(f"{arm}/{method}")
                if complete and ablated and complete_partners(primary, complete) and complete_partners(primary, ablated):
                    summary["ablations"][f"{arm}/{method}"] = paired_bootstrap_difference(mean_over_members(complete), mean_over_members(ablated), primary)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(_rounded(summary), f, indent=1)
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
