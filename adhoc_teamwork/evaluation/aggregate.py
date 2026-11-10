"""Turns benchmark tables into the comparison the proposal asks for (Section 6:
"evaluate on held-out biased partners and multiple random seeds and compare
average cooperative returns across baselines").

Uncertainty comes from two places, handled separately:

* Across training seeds or baseline seeds: each method family's score is the
  mean over its seeds, and the spread is reported.
* Across partners: a difference between two methods is only meaningful if it
  holds across partners rather than coming from one or two of them, so
  differences are bootstrapped over partners, pairing the two methods on each.

Each difference also gets a p-value from an exact sign-flip test over the same
distinct partners. A table makes many comparisons at once, so ``holm_adjust``
corrects a table's p-values for that (Holm, 1979).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np

# method name -> family, e.g. traj_s36_3 -> traj_s36, mappo_sp2_final -> mappo.
_FAMILY_RULES = [
    (re.compile(r"^mappo_"), "mappo"),
    (re.compile(r"^sub_"), "single_sub_policy"),
    (re.compile(r"^subargmax_"), "single_sub_policy_argmax"),
    (re.compile(r"^(traj|hsp|fcp|mep|cole)_s\d+_\d+$"), None),
    (re.compile(r"^e3t_\d+$"), "e3t"),
]

BASELINE_FAMILIES = ("mappo", "traj_s36", "hsp_s36", "fcp_s36", "mep_s36", "cole_s75", "e3t")


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


def partner_cluster(partner: str) -> str:
    """``bias3_mid`` and ``bias3_final`` are two checkpoints of one agent."""
    return re.sub(r"_(init|mid|final)$", "", partner)


def sign_flip_p_value(differences: np.ndarray, n_resamples: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value for a mean difference of zero, flipping the sign of
    each independent difference at random: exact (every sign pattern) for up to
    16 differences, Monte Carlo above. With n differences it is never below
    2 / 2**n, so 5 partners can never reach 0.05.
    """
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n <= 16:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_resamples, n))
        signs[0] = 1  # include the observed pattern
    flipped = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(flipped >= abs(differences.mean()) - 1e-9))


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-adjusted p-values for one family of comparisons: a comparison is
    significant at level alpha, controlling the chance of any false positive in
    the family, when its adjusted p-value is at most alpha.
    """
    adjusted, running = {}, 0.0
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[key] = running
    return adjusted


def paired_bootstrap_difference(a: Dict[str, float], b: Dict[str, float], partners: Sequence[str], n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Mean of a - b over partners, with a percentile bootstrap CI and a
    sign-flip p-value. Both resample distinct agents rather than checkpoints,
    which are not independent and would make the interval too narrow.
    """
    clusters: Dict[str, List[float]] = {}
    for p in partners:
        clusters.setdefault(partner_cluster(p), []).append(a[p] - b[p])
    cluster_diffs = np.array([np.mean(v) for v in clusters.values()])
    rng = np.random.default_rng(seed)
    resampled = cluster_diffs[rng.integers(0, len(cluster_diffs), size=(n_boot, len(cluster_diffs)))].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    diffs = np.array([a[p] - b[p] for p in partners])
    return {
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "partners_better": int((diffs > 0).sum()),
        "n_partners": int(len(diffs)),
        "n_independent_partners": int(len(clusters)),
        "p_value": sign_flip_p_value(cluster_diffs),
    }


def mean_over_members(scores: Dict[str, List[float]]) -> Dict[str, float]:
    return {p: float(np.mean(v)) for p, v in scores.items()}


def population_steps(scores: Dict[str, Dict[str, List[float]]], partners: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Paired differences that need no trained network: what ensembling and
    argmax each contribute, and whether argmax helps ensembles more than it
    helps their members. Steps whose families have no results are left out.
    """
    core = ("single_sub_policy", "single_sub_policy_argmax", "static_ensemble", "static_ensemble_argmax")
    families = core + ("static_ensemble_vote",)
    means = {f: mean_over_members(scores[f]) for f in families if f in scores and all(p in scores[f] for p in partners)}
    steps = {}
    for name, a, b in (
        ("ensembling_sampled", "static_ensemble", "single_sub_policy"),
        ("ensembling_argmax", "static_ensemble_argmax", "single_sub_policy_argmax"),
        ("argmax_on_ensemble", "static_ensemble_argmax", "static_ensemble"),
        ("argmax_on_single", "single_sub_policy_argmax", "single_sub_policy"),
        ("vote_vs_sampled", "static_ensemble_vote", "static_ensemble"),
        ("vote_vs_mean_argmax", "static_ensemble_vote", "static_ensemble_argmax"),
    ):
        if a in means and b in means:
            steps[name] = paired_bootstrap_difference(means[a], means[b], partners)
    if all(f in means for f in core):
        on_ensemble = {p: means["static_ensemble_argmax"][p] - means["static_ensemble"][p] for p in partners}
        on_single = {p: means["single_sub_policy_argmax"][p] - means["single_sub_policy"][p] for p in partners}
        steps["argmax_interaction"] = paired_bootstrap_difference(on_ensemble, on_single, partners)
    return steps


def uniform_vs_baselines(scores: Dict[str, Dict[str, List[float]]], partners: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """The uniform ensemble minus each baseline, like for like: sampled against
    sampled, argmax against argmax. Needs no trained network.
    """
    result = {}
    for ensemble, suffix in (("static_ensemble", ""), ("static_ensemble_argmax", "_argmax")):
        if ensemble not in scores or not all(p in scores[ensemble] for p in partners):
            continue
        ours = mean_over_members(scores[ensemble])
        result[ensemble] = {
            f"{family}{suffix}": paired_bootstrap_difference(ours, mean_over_members(scores[f"{family}{suffix}"]), partners)
            for family in BASELINE_FAMILIES
            if f"{family}{suffix}" in scores and all(p in scores[f"{family}{suffix}"] for p in partners)
        }
    return result
