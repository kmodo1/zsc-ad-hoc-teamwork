#!/usr/bin/env python
"""Print the README's result tables from ``results/overcooked/<layout>/<population>/summary.json``
(written by ``slurm/analyze_all.sh``).

Each cell that compares two methods is a mean paired difference over a group's
partners, with a 95% bootstrap CI over the distinct partner agents. Each table
is one family of comparisons: its p-values (exact sign-flip tests over the same
agents) are Holm-adjusted together, and a cell marked † is significant at 0.05
after that correction. Cells whose runs have not finished are shown as "-".

Usage:
    python -m adhoc_teamwork.scripts.report_tables [--results adhoc_teamwork/results/overcooked] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from adhoc_teamwork.evaluation.aggregate import holm_adjust, paired_bootstrap_difference

LAYOUTS = ("unident_s", "random0_medium", "random3")
# The training partners of each population's main results.
MAIN_ARMS = {"traj-full10": "actor_critic-sp", "s2-mix": "actor_critic-sp+biased", "hsp-w1": "actor_critic-sp+biased"}
GROUP_NAMES = {"biased": "Biased", "mep": "MEP", "self_play": "Self-play", "traj": "TrajeDi", "cole": "COLE", "e3t": "E3T"}
# Which partner group holds each training generator's unseen agents.
GENERATOR_GROUPS = {"sp": "self_play", "biased": "biased", "mep": "mep", "traj": "traj"}
MIXES = ("s2-mix", "mix-g8s4", "mix-g4s8", "mix-g4s4t4", "hsp-w1")
MIX_LABELS = {"s2-mix": "12 : 0 : 0", "mix-g8s4": "8 : 4 : 0", "mix-g4s8": "4 : 8 : 0", "mix-g4s4t4": "4 : 4 : 4", "hsp-w1": "0 : 10 : 0"}


class Results:
    def __init__(self, root: str):
        self.root = root
        self._cache: Dict[tuple, Optional[dict]] = {}

    def summary(self, layout: str, population: str) -> Optional[dict]:
        key = (layout, population)
        if key not in self._cache:
            path = os.path.join(self.root, layout, population, "summary.json")
            self._cache[key] = json.load(open(path)) if os.path.exists(path) else None
        return self._cache[key]

    def partners(self, layout: str, population: str, group: str) -> Optional[List[str]]:
        s = self.summary(layout, population)
        return s["partner_groups"].get(group) if s and "partner_groups" in s else None

    def scores(self, layout: str, population: str, family: str, group: str) -> Optional[Dict[str, float]]:
        """{partner: mean over seeds}, or None unless every partner of the group has one."""
        partners = self.partners(layout, population, group)
        means = (self.summary(layout, population) or {}).get("partner_means", {}).get(family)
        if partners is None or means is None or not all(p in means for p in partners):
            return None
        return {p: means[p] for p in partners}

    def mean(self, layout: str, population: str, family: str, group: str = "biased") -> Optional[float]:
        scores = self.scores(layout, population, family, group)
        return None if scores is None else sum(scores.values()) / len(scores)

    def best_baseline(self, layout: str, population: str, group: str, argmax: bool) -> Optional[str]:
        s = self.summary(layout, population)
        if not s or group not in s.get("best_baselines", {}):
            return None
        return s["best_baselines"][group]["best_baseline_argmax" if argmax else "best_baseline"]

    def difference(self, layout: str, population: str, a: str, b: str, group: str = "biased", population_b: str = None) -> Optional[dict]:
        """a - b on one group; b may come from another population's file (same partners)."""
        sa = self.scores(layout, population, a, group)
        sb = self.scores(layout, population_b or population, b, group)
        if sa is None or sb is None or sorted(sa) != sorted(sb):
            return None
        return paired_bootstrap_difference(sa, sb, sorted(sa))


class Table:
    """A markdown table whose difference cells are Holm-adjusted together."""

    def __init__(self, title: str, header: Sequence[str], note: str = ""):
        self.title, self.header, self.note = title, list(header), note
        self.rows: List[list] = []

    def add(self, *cells) -> None:
        self.rows.append(list(cells))

    def render(self) -> str:
        diffs = {(i, j): c for i, row in enumerate(self.rows) for j, c in enumerate(row) if isinstance(c, dict)}
        adjusted = holm_adjust({k: c["p_value"] for k, c in diffs.items()})
        lines = [f"### {self.title}", ""]
        if self.note:
            lines += [self.note, ""]
        lines += ["| " + " | ".join(self.header) + " |", "|" + "|".join("---" if j == 0 else "---:" for j in range(len(self.header))) + "|"]
        for i, row in enumerate(self.rows):
            cells = []
            for j, c in enumerate(row):
                if isinstance(c, dict):
                    mark = " †" if adjusted[(i, j)] <= 0.05 else ""
                    cells.append(f"{c['mean_difference']:+.1f} [{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}]{mark}")
                elif isinstance(c, float):
                    cells.append(f"{c:.1f}")
                else:
                    cells.append("-" if c is None else str(c))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"{len(diffs)} comparisons; † = significant at 0.05 after Holm correction across this table.")
        return "\n".join(lines) + "\n"


def code(s: str) -> str:
    return f"`{s}`"


def adaptive_vs_uniform(r: Results) -> Table:
    t = Table("Adaptive against uniform weights, same population", ["Layout", "Population", "Sampled", "Argmax"])
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            t.add(
                code(layout),
                code(population),
                r.difference(layout, population, f"{arm}/adaptive_ensemble", "static_ensemble"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble_argmax", "static_ensemble_argmax"),
            )
    return t


def adaptive_vs_best_baseline(r: Results) -> Table:
    t = Table("Adaptive against the best baseline, like for like", ["Layout", "Population", "Sampled", "Argmax"])
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            cells = []
            for method, argmax in (("adaptive_ensemble", False), ("adaptive_ensemble_argmax", True)):
                best = r.best_baseline(layout, population, "biased", argmax)
                cells.append(r.difference(layout, population, f"{arm}/{method}", best) if best else None)
            t.add(code(layout), code(population), *cells)
    return t


def encoder_ablations(r: Results) -> Table:
    t = Table(
        "Encoder ablations (complete encoder minus ablated one)",
        ["Layout", "Population", "No memory, sampled", "No memory, argmax", "Partner actions hidden, sampled", "Partner actions hidden, argmax"],
    )
    for layout in LAYOUTS:
        for population, arm in (("hsp-w1", "actor_critic-sp+biased"), ("s2-mix", "actor_critic-sp+biased"), ("traj-full10", "actor_critic-sp")):
            cells = [
                r.difference(layout, population, f"{arm}/{method}", f"{arm}-{ablation}/{method}")
                for ablation in ("memoryless", "no_partner_action")
                for method in ("adaptive_ensemble", "adaptive_ensemble_argmax")
            ]
            if any(cells):
                t.add(code(layout), code(population), *[c if c else "not run" for c in cells])
    return t


def training_partners(r: Results) -> Table:
    t = Table("Training partners: `hsp-w1` against uniform weights", ["Layout", "Trained against", "Sampled", "Argmax"])
    for layout in LAYOUTS:
        for arm, label in (("actor_critic-sp", "Self-play only"), ("actor_critic-sp+biased", "Self-play + other biased agents")):
            t.add(
                code(layout),
                label,
                r.difference(layout, "hsp-w1", f"{arm}/adaptive_ensemble", "static_ensemble"),
                r.difference(layout, "hsp-w1", f"{arm}/adaptive_ensemble_argmax", "static_ensemble_argmax"),
            )
    return t


def other_groups(r: Results) -> Table:
    t = Table(
        "Other partner groups, argmax actions",
        ["Layout", "Population", "Partners", "Adaptive", "Against uniform", "Best baseline", "Against best baseline"],
        "Each population's main training run. A group is left out where it overlaps the population or what its members "
        "trained with, and a baseline is left out of a group it trained with.",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            s = r.summary(layout, population)
            if not s:
                continue
            for group in s["partner_groups"]:
                if group == "biased":
                    continue
                ours = f"{arm}/adaptive_ensemble_argmax"
                best = r.best_baseline(layout, population, group, argmax=True)
                t.add(
                    code(layout),
                    code(population),
                    GROUP_NAMES[group],
                    r.mean(layout, population, ours, group),
                    r.difference(layout, population, ours, "static_ensemble_argmax", group),
                    f"{best.removesuffix('_argmax')} {r.mean(layout, population, best, group):.1f}" if best and r.mean(layout, population, best, group) is not None else None,
                    r.difference(layout, population, ours, best, group) if best else None,
                )
    return t


def generator_transfer(r: Results) -> List[Table]:
    """Does what the network learned carry over to a generator it never trained with?"""
    returns = Table(
        "Generator held out: `hsp-w1`, argmax actions, mean return",
        ["Layout", "Trained against"] + [GROUP_NAMES[g] for g in GROUP_NAMES],
        "Rows train against different generators; each column is a group of unseen partners. Training on all four "
        "generators balances them; the original two sets sample agents uniformly.",
    )
    arms = [("actor_critic-sp", "Self-play"), ("actor_critic-sp+biased", "Self-play + biased"), ("actor_critic-all-gen", "All four")]
    arms += [(f"actor_critic-no-{g}", f"All but {GROUP_NAMES[GENERATOR_GROUPS[g]].lower() if g != 'sp' else 'self-play'}") for g in GENERATOR_GROUPS]
    for layout in LAYOUTS:
        returns.add(code(layout), "*Uniform weights*", *[r.mean(layout, "hsp-w1", "static_ensemble_argmax", g) for g in GROUP_NAMES])
        for arm, label in arms:
            returns.add(code(layout), label, *[r.mean(layout, "hsp-w1", f"{arm}/adaptive_ensemble_argmax", g) for g in GROUP_NAMES])
    gaps = Table(
        "Generator held out: the cost of not having trained with it",
        ["Layout", "Held-out generator", "Trained with it", "Trained without it", "With minus without", "Without, against uniform"],
        "`hsp-w1`, argmax actions, on the unseen agents of the held-out generator.",
    )
    for layout in LAYOUTS:
        for generator, group in GENERATOR_GROUPS.items():
            with_it, without = "actor_critic-all-gen/adaptive_ensemble_argmax", f"actor_critic-no-{generator}/adaptive_ensemble_argmax"
            gaps.add(
                code(layout),
                GROUP_NAMES[group],
                r.mean(layout, "hsp-w1", with_it, group),
                r.mean(layout, "hsp-w1", without, group),
                r.difference(layout, "hsp-w1", with_it, without, group),
                r.difference(layout, "hsp-w1", without, "static_ensemble_argmax", group),
            )
    return [returns, gaps]


def action_rules(r: Results) -> Table:
    t = Table(
        "Training the way it acts",
        ["Layout", "Population", "Ensemble, argmax", "Policy selection, sampled", "Policy selection, argmax"],
        "Each cell: trained with the evaluation's action rule minus trained by sampling from the mixture, both evaluated "
        "the same way, on the biased partners.",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            t.add(
                code(layout),
                code(population),
                *[
                    r.difference(layout, population, f"{arm}-{rule}/{method}", f"{arm}/{method}")
                    for rule, method in (("argmax", "adaptive_ensemble_argmax"), ("select", "policy_selection"), ("select_argmax", "policy_selection_argmax"))
                ],
            )
    return t


def action_rule_returns(r: Results) -> Table:
    t = Table(
        "Training the way it acts: mean return on the biased partners",
        ["Layout", "Population", "Uniform, argmax", "Ensemble, argmax (trained sampling / argmax)", "Policy selection, argmax (trained sampling / selecting)"],
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            def pair(a, b):
                va, vb = r.mean(layout, population, a), r.mean(layout, population, b)
                return None if va is None or vb is None else f"{va:.1f} / {vb:.1f}"

            t.add(
                code(layout),
                code(population),
                r.mean(layout, population, "static_ensemble_argmax"),
                pair(f"{arm}/adaptive_ensemble_argmax", f"{arm}-argmax/adaptive_ensemble_argmax"),
                pair(f"{arm}/policy_selection_argmax", f"{arm}-select_argmax/policy_selection_argmax"),
            )
    return t


def policy_selection(r: Results) -> Table:
    t = Table(
        "Policy selection against the adaptive ensemble, argmax actions",
        ["Layout", "Population", "Selection, trained sampling", "Selection, trained selecting", "Against the ensemble (both trained their own way)"],
        "Mean return on the biased partners, then the selection agent trained by selecting minus the ensemble trained with argmax.",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            t.add(
                code(layout),
                code(population),
                r.mean(layout, population, f"{arm}/policy_selection_argmax"),
                r.mean(layout, population, f"{arm}-select_argmax/policy_selection_argmax"),
                r.difference(layout, population, f"{arm}-select_argmax/policy_selection_argmax", f"{arm}-argmax/adaptive_ensemble_argmax"),
            )
    return t


def mixed_populations(r: Results) -> Table:
    t = Table(
        "Mixed populations, argmax actions, biased partners",
        ["Layout", "Generalists : specialists : TrajeDi", "Uniform", "Adaptive (trained sampling)", "Adaptive (trained argmax)", "Adaptive (sampling) against uniform", "Adaptive (sampling) against best baseline"],
    )
    for layout in LAYOUTS:
        for population in MIXES:
            arm = "actor_critic-sp+biased"
            best = r.best_baseline(layout, population, "biased", argmax=True)
            t.add(
                code(layout),
                MIX_LABELS[population],
                r.mean(layout, population, "static_ensemble_argmax"),
                r.mean(layout, population, f"{arm}/adaptive_ensemble_argmax"),
                r.mean(layout, population, f"{arm}-argmax/adaptive_ensemble_argmax"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble_argmax", "static_ensemble_argmax"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble_argmax", best) if best else None,
            )
    return t


def reinforce(r: Results) -> Table:
    t = Table(
        "Actor-critic against REINFORCE",
        ["Layout", "Population", "REINFORCE, sampled", "Actor-critic minus REINFORCE, sampled", "REINFORCE, argmax", "Actor-critic minus REINFORCE, argmax"],
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            other = arm.replace("actor_critic", "reinforce")
            t.add(
                code(layout),
                code(population),
                r.mean(layout, population, f"{other}/adaptive_ensemble"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble", f"{other}/adaptive_ensemble"),
                r.mean(layout, population, f"{other}/adaptive_ensemble_argmax"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble_argmax", f"{other}/adaptive_ensemble_argmax"),
            )
    return t


TABLES = {
    "adaptive_vs_uniform": adaptive_vs_uniform,
    "adaptive_vs_best_baseline": adaptive_vs_best_baseline,
    "encoder_ablations": encoder_ablations,
    "training_partners": training_partners,
    "other_groups": other_groups,
    "generator_transfer": generator_transfer,
    "action_rules": action_rules,
    "action_rule_returns": action_rule_returns,
    "policy_selection": policy_selection,
    "mixed_populations": mixed_populations,
    "reinforce": reinforce,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="adhoc_teamwork/results/overcooked")
    parser.add_argument("--only", nargs="+", choices=list(TABLES), default=None)
    args = parser.parse_args()
    results = Results(args.results)
    for name in args.only or TABLES:
        made = TABLES[name](results)
        for table in made if isinstance(made, list) else [made]:
            print(table.render())


if __name__ == "__main__":
    main()
