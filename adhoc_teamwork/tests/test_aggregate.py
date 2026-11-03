"""Unit tests for the evaluation aggregation and its resumable result files."""

import json

import numpy as np

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.envs.vec_env import VecEnv
from adhoc_teamwork.evaluation.aggregate import (
    best_fixed_sub_policy,
    family_partner_scores,
    group_summary,
    method_family,
    oracle_best_sub_policy,
    paired_bootstrap_difference,
)
from adhoc_teamwork.evaluation.benchmark import _load_done, method_partner_means, run_benchmark
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent


def test_method_family_groups_seeds_of_one_algorithm():
    assert method_family("traj_s36_3") == "traj_s36"
    assert method_family("mappo_sp2_final") == "mappo"
    assert method_family("sub_traj4_final") == "single_sub_policy"
    assert method_family("static_ensemble") == "static_ensemble"


def test_family_scores_and_summary():
    tables = [{"traj_s36_1": {"p1": 10.0, "p2": 20.0}, "traj_s36_2": {"p1": 30.0, "p2": 40.0}}]
    scores = family_partner_scores(tables)
    assert scores["traj_s36"] == {"p1": [10.0, 30.0], "p2": [20.0, 40.0]}
    summary = group_summary(scores["traj_s36"], ["p1", "p2"])
    assert summary["mean"] == 25.0
    assert summary["n_seeds"] == 2
    assert np.isclose(summary["std_across_seeds"], np.std([15.0, 35.0], ddof=1))


def test_oracle_takes_best_sub_policy_per_partner_and_ignores_other_methods():
    table = {"sub_a": {"p1": 10.0, "p2": 0.0}, "sub_b": {"p1": 0.0, "p2": 5.0}, "static_ensemble": {"p1": 99.0, "p2": 99.0}}
    assert oracle_best_sub_policy(table) == {"p1": 10.0, "p2": 5.0}


def test_best_fixed_sub_policy_is_one_policy_not_a_per_partner_max():
    table = {"sub_a": {"p1": 10.0, "p2": 0.0}, "sub_b": {"p1": 6.0, "p2": 6.0}, "static_ensemble": {"p1": 99.0, "p2": 99.0}}
    assert best_fixed_sub_policy(table, ["p1", "p2"]) == {"p1": 6.0, "p2": 6.0}


def test_paired_bootstrap_detects_consistent_difference_and_not_a_lucky_one():
    partners = [f"p{i}" for i in range(20)]
    base = {p: float(i * 10) for i, p in enumerate(partners)}
    consistent = paired_bootstrap_difference({p: v + 5.0 for p, v in base.items()}, base, partners)
    assert consistent["ci95_low"] == consistent["ci95_high"] == consistent["mean_difference"] == 5.0
    assert consistent["partners_better"] == 20

    one_partner = dict(base)
    one_partner["p0"] += 100.0
    lucky = paired_bootstrap_difference(one_partner, base, partners)
    assert lucky["mean_difference"] == 5.0
    assert lucky["ci95_low"] == 0.0, "a difference from one partner must not look significant"


def test_benchmark_covers_both_positions_and_resumes(tmp_path):
    vec_env = VecEnv(lambda: OvercookedGridworld(horizon=5), 2)
    out = str(tmp_path / "results.jsonl")
    methods = {"m1": lambda: ScriptedBiasedAgent("pot_tender"), "m2": lambda: ScriptedBiasedAgent("onion_runner")}
    partners = {"p1": lambda: ScriptedBiasedAgent("random_noise")}
    rows = run_benchmark(vec_env, methods, partners, out, verbose=False)
    assert len(rows) == 4
    assert {(r["method"], r["position"]) for r in rows} == {("m1", 0), ("m1", 1), ("m2", 0), ("m2", 1)}
    assert all(len(r["returns"]) == 2 for r in rows)

    # A line truncated by preemption is ignored, and only that cell is redone.
    with open(out) as f:
        lines = f.readlines()
    with open(out, "w") as f:
        f.writelines(lines[:-1] + [lines[-1][:10]])
    assert len(_load_done(out)) == 3
    rows = run_benchmark(vec_env, methods, partners, out, verbose=False)
    assert len(rows) == 4
    with open(out) as f:
        parsed = []
        for line in f:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    assert len(parsed) == 4, "finished cells must not be run again"
    assert set(method_partner_means(rows)) == {"m1", "m2"}


def test_partially_finished_families_are_left_out_of_analysis():
    from adhoc_teamwork.scripts.analyze_overcooked import complete_partners

    assert complete_partners(["p1", "p2"], {"p1": [1.0, 2.0], "p2": [3.0, 4.0]})
    assert not complete_partners(["p1", "p2"], {"p1": [1.0]})
    assert not complete_partners(["p1", "p2"], {"p1": [1.0, 2.0], "p2": [3.0]})
