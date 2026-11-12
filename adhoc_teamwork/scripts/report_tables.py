#!/usr/bin/env python
"""Print the README's result tables from ``results/overcooked/<layout>/<population>/summary.json``
(written by ``slurm/analyze_all.sh``).

Each cell that compares two methods is a mean paired difference over a group's
partners, with a 95% bootstrap CI over the distinct partner agents. Each column
of a table asks one question across layouts and populations, so it is one family
of comparisons: its p-values (exact sign-flip tests over the same agents) are
Holm-adjusted together, and a cell marked † is significant at 0.05 after that
correction. With 10 agents no p-value is below 2/1024, so a whole table as one
family could leave nothing significant by construction. Cells whose runs have
not finished are shown as "-".

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

    def partners_by_group(self, layout: str, population: str) -> Dict[str, List[str]]:
        s = self.summary(layout, population)
        return s.get("partner_groups", {}) if s else {}

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
    """A markdown table whose difference cells are Holm-adjusted within each column."""

    def __init__(self, title: str, header: Sequence[str], note: str = "", text_columns: int = 2):
        self.title, self.header, self.note, self.text_columns = title, list(header), note, text_columns
        self.rows: List[list] = []

    def add(self, *cells) -> None:
        self.rows.append(list(cells))

    def render(self) -> str:
        diffs = {(i, j): c for i, row in enumerate(self.rows) for j, c in enumerate(row) if isinstance(c, dict)}
        adjusted = {}
        for column in {j for _, j in diffs}:
            adjusted.update(holm_adjust({k: c["p_value"] for k, c in diffs.items() if k[1] == column}))
        lines = [f"### {self.title}", ""]
        if self.note:
            lines += [self.note, ""]
        lines += ["| " + " | ".join(self.header) + " |", "|" + "|".join("---" if j < self.text_columns else "---:" for j in range(len(self.header))) + "|"]
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
        if diffs:
            columns = len({j for _, j in diffs})
            lines.append(f"† = significant at 0.05 after Holm correction within its column ({len(diffs)} comparisons in {columns} column{'s' if columns > 1 else ''}).")
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


OTHER_GROUPS = ("mep", "self_play", "traj", "cole", "e3t")


def other_groups(r: Results) -> List[Table]:
    """The main runs on every held-out group but the biased one, argmax actions."""
    note = (
        "Each population's main training run, argmax actions. \"n/a\": the group overlaps the population or what its "
        "members trained with."
    )
    against_uniform = Table("Other partner groups: adaptive against uniform weights", ["Layout", "Population"] + [GROUP_NAMES[g] for g in OTHER_GROUPS], note)
    against_best = Table(
        "Other partner groups: adaptive against the best baseline",
        ["Layout", "Population"] + [GROUP_NAMES[g] for g in OTHER_GROUPS],
        note + " Baselines that trained with a group's agents are left out of it (MAPPO and FCP for self-play, TrajeDi "
        "for TrajeDi, E3T for E3T).",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            groups = r.partners_by_group(layout, population)
            ours = f"{arm}/adaptive_ensemble_argmax"
            uniform_cells, best_cells = [], []
            for g in OTHER_GROUPS:
                if g not in groups:
                    uniform_cells.append("n/a")
                    best_cells.append("n/a")
                    continue
                best = r.best_baseline(layout, population, g, argmax=True)
                uniform_cells.append(r.difference(layout, population, ours, "static_ensemble_argmax", g))
                best_cells.append(r.difference(layout, population, ours, best, g) if best else None)
            against_uniform.add(code(layout), code(population), *uniform_cells)
            against_best.add(code(layout), code(population), *best_cells)
    return [against_uniform, against_best]


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
        ["Layout", "Population", "Ensemble, argmax (noisy weights)", "Ensemble, argmax (annealed temperature)", "Policy selection, sampled", "Policy selection, argmax"],
        "Each cell: trained with the evaluation's action rule minus trained by sampling from the mixture, both evaluated "
        "the same way, on the biased partners. The annealed rule was run on `s2-mix` only, on two layouts.",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            t.add(
                code(layout),
                code(population),
                *[
                    r.difference(layout, population, f"{arm}-{rule}/{method}", f"{arm}/{method}")
                    for rule, method in (
                        ("argmax", "adaptive_ensemble_argmax"),
                        ("anneal", "adaptive_ensemble_argmax"),
                        ("select", "policy_selection"),
                        ("select_argmax", "policy_selection_argmax"),
                    )
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
        ["Layout", "Population", "Selection", "Ensemble", "Ensemble minus selection"],
        "Mean return on the biased partners. Both use the same trained network (each population's main run); selection "
        "acts with the highest-weighted sub-policy, the ensemble with the weighted mixture.",
    )
    for layout in LAYOUTS:
        for population, arm in MAIN_ARMS.items():
            t.add(
                code(layout),
                code(population),
                r.mean(layout, population, f"{arm}/policy_selection_argmax"),
                r.mean(layout, population, f"{arm}/adaptive_ensemble_argmax"),
                r.difference(layout, population, f"{arm}/adaptive_ensemble_argmax", f"{arm}/policy_selection_argmax"),
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


def mixed_populations_other_groups(r: Results) -> Table:
    t = Table(
        "Mixed populations on unseen generators, argmax actions",
        ["Layout", "Generalists : specialists : TrajeDi", "MEP", "COLE", "E3T", "Mean against `s2-mix`"],
        "Mean return of the adaptive ensemble (trained by sampling) on the groups every mix is held out from, then the "
        "difference from `s2-mix` over all their partners together.",
    )
    groups = ("mep", "cole", "e3t")
    for layout in LAYOUTS:
        for population in MIXES:
            arm = "actor_critic-sp+biased"
            ours = f"{arm}/adaptive_ensemble_argmax"
            pooled = {}
            for g in groups:
                a, b = r.scores(layout, population, ours, g), r.scores(layout, "s2-mix", ours, g)
                if a is None or b is None:
                    pooled = None
                    break
                pooled.update({p: (a[p], b[p]) for p in a})
            diff = paired_bootstrap_difference({p: v[0] for p, v in pooled.items()}, {p: v[1] for p, v in pooled.items()}, sorted(pooled)) if pooled and population != "s2-mix" else None
            t.add(code(layout), MIX_LABELS[population], *[r.mean(layout, population, ours, g) for g in groups], diff)
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
    "mixed_populations_other_groups": mixed_populations_other_groups,
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
