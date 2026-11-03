"""Turns benchmark tables into the comparison the proposal asks for (Section 6:
"evaluate on held-out biased partners and multiple random seeds and compare
average cooperative returns across baselines").

Uncertainty comes from two places, handled separately:

* Across training seeds or baseline seeds: each method family's score is the
  mean over its seeds, and the spread is reported.
* Across partners: a difference between two methods is only meaningful if it
  holds across partners rather than coming from one or two of them, so
  differences are bootstrapped over partners, pairing the two methods on each.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np

# method name -> family, e.g. traj_s36_3 -> traj_s36, mappo_sp2_final -> mappo.
_FAMILY_RULES = [
    (re.compile(r"^mappo_"), "mappo"),
    (re.compile(r"^sub_"), "single_sub_policy"),
    (re.compile(r"^(traj|hsp|fcp|mep|cole)_s\d+_\d+$"), None),
]


def method_family(method: str) -> str:
    for pattern, family in _FAMILY_RULES:
        if pattern.match(method):
            return family if family is not None else method.rsplit("_", 1)[0]
    return method


def family_partner_scores(tables: Iterable[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, List[float]]]:
    """Group {method: {partner: mean return}} tables by method family. Returns
    {family: {partner: [score of each family member / training seed]}}.
    """
    scores: Dict[str, Dict[str, List[float]]] = {}
    for table in tables:
        for method, partners in table.items():
            family = scores.setdefault(method_family(method), {})
            for partner, value in partners.items():
                family.setdefault(partner, []).append(value)
    return scores


def oracle_best_sub_policy(table: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """For each partner, the return of the best single sub-policy for it, in
    hindsight. An estimate of what perfect per-partner selection could reach
    (slightly optimistic, as the max of noisy means is biased upward).
    """
    subs = {m: p for m, p in table.items() if m.startswith("sub_")}
    partners = next(iter(subs.values())).keys()
    return {partner: max(p[partner] for p in subs.values()) for partner in partners}


def best_fixed_sub_policy(table: Dict[str, Dict[str, float]], partners: Sequence[str]) -> Dict[str, float]:
    """The one sub-policy with the best mean over ``partners``, in hindsight:
    what could be reached with no partner adaptation at all. Anything claiming
    a benefit from adapting to the partner has to beat this, not just the
    uniform mixture.
    """
    subs = {m: p for m, p in table.items() if m.startswith("sub_")}
    best = max(subs, key=lambda m: np.mean([subs[m][p] for p in partners]))
    return dict(subs[best])


def group_summary(scores: Dict[str, List[float]], partners: Sequence[str]) -> Dict[str, float]:
    """Mean over ``partners`` of a family's mean score, plus the spread of
    per-seed means. All members must have results for all partners."""
    per_partner = np.array([scores[p] for p in partners])  # (partners, members)
    per_member = per_partner.mean(axis=0)
    return {
        "mean": float(per_member.mean()),
        "std_across_seeds": float(per_member.std(ddof=1)) if per_member.size > 1 else 0.0,
        "n_seeds": int(per_member.size),
    }


def paired_bootstrap_difference(a: Dict[str, float], b: Dict[str, float], partners: Sequence[str], n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Mean of a - b over partners, with a percentile bootstrap CI over partners."""
    diffs = np.array([a[p] - b[p] for p in partners])
    rng = np.random.default_rng(seed)
    resampled = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "partners_better": int((diffs > 0).sum()),
        "n_partners": int(len(diffs)),
    }


def mean_over_members(scores: Dict[str, List[float]]) -> Dict[str, float]:
    return {p: float(np.mean(v)) for p, v in scores.items()}
