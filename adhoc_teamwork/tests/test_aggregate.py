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
    population_steps,
)
from adhoc_teamwork.evaluation.benchmark import _load_done, method_partner_means, run_benchmark
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent


def test_method_family_groups_seeds_of_one_algorithm():
    assert method_family("traj_s36_3") == "traj_s36"
    assert method_family("mappo_sp2_final") == "mappo"
    assert method_family("sub_traj4_final") == "single_sub_policy"
    assert method_family("subargmax_traj4_final") == "single_sub_policy_argmax"
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
    table = {"sub_a": {"p1": 10.0, "p2": 0.0}, "sub_b": {"p1": 0.0, "p2": 5.0}, "subargmax_a": {"p1": 99.0, "p2": 99.0}, "static_ensemble": {"p1": 99.0, "p2": 99.0}}
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


def test_bootstrap_resamples_agents_not_their_checkpoints():
    # Two agents, one with a large difference at both checkpoints. Treating the
    # four checkpoints as independent would understate the uncertainty.
    partners = ["bias1_mid", "bias1_final", "bias2_mid", "bias2_final"]
    a = {"bias1_mid": 10.0, "bias1_final": 10.0, "bias2_mid": 0.0, "bias2_final": 0.0}
    b = {p: 0.0 for p in partners}
    result = paired_bootstrap_difference(a, b, partners)
    assert result["n_independent_partners"] == 2
    assert result["n_partners"] == 4
    assert result["ci95_low"] == 0.0 and result["ci95_high"] == 10.0


def test_population_steps_separate_argmax_from_ensembling():
    partners = ["bias1_mid", "bias1_final", "bias2_mid", "bias2_final"]

    def family(*member_values):
        return {p: list(member_values) for p in partners}

    scores = {
        "single_sub_policy": family(100.0, 120.0),  # mean 110
        "single_sub_policy_argmax": family(110.0, 130.0),  # mean 120
        "static_ensemble": family(130.0),
        "static_ensemble_argmax": family(170.0),
    }
    steps = population_steps(scores, partners)
    assert steps["ensembling_sampled"]["mean_difference"] == 20.0
    assert steps["ensembling_argmax"]["mean_difference"] == 50.0
    assert steps["argmax_on_ensemble"]["mean_difference"] == 40.0
    assert steps["argmax_on_single"]["mean_difference"] == 10.0
    assert steps["argmax_interaction"]["mean_difference"] == 30.0


def test_population_steps_skip_families_without_results():
    partners = ["p1", "p2"]
    scores = {"single_sub_policy": {"p1": [1.0], "p2": [2.0]}, "static_ensemble": {"p1": [3.0], "p2": [4.0]}, "static_ensemble_argmax": {"p1": [5.0]}}
    assert set(population_steps(scores, partners)) == {"ensembling_sampled"}


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


def test_gain_breakdown_reports_missing_memoryless_runs_as_not_run():
    from adhoc_teamwork.scripts.gain_breakdown import fmt, weighting_steps

    step = {"mean_difference": 4.0, "ci95_low": 1.0, "ci95_high": 7.0}
    summary = {
        "comparisons": {
            "arm/adaptive_ensemble_argmax": {"static_ensemble_argmax": step},
            "arm-memoryless/adaptive_ensemble_argmax": {"static_ensemble_argmax": step},
            "other/adaptive_ensemble": {"static_ensemble": step},
        },
        "ablations": {"arm-memoryless/adaptive_ensemble_argmax": step},
    }
    complete = weighting_steps(summary, "arm", "adaptive_ensemble_argmax")
    assert complete == {"state_weights": step, "memory": step, "all_weights": step}
    missing = weighting_steps(summary, "other", "adaptive_ensemble")
    assert missing == {"state_weights": None, "memory": None, "all_weights": step}
    assert fmt(step) == "+4.0 [+1.0, +7.0]"
    assert fmt(None) == "not run"
