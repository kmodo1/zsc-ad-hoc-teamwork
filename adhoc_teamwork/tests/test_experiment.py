"""The train/evaluation partner split is what makes "unseen partner" mean
anything (Section 6, Section 9), so it is tested rather than assumed."""

import pytest

pytest.importorskip("zsceval")

from adhoc_teamwork import experiment  # noqa: E402


# Populations that read the pool to choose their members.
_POOL_POPULATIONS = ("hsp-w1", "hsp-w1-new") + tuple(experiment.MIXED_POPULATIONS)


@pytest.mark.parametrize("layout", ["unident_s", "random0_medium"])
@pytest.mark.parametrize("population", [p for p in experiment.POPULATIONS if p not in _POOL_POPULATIONS])
def test_evaluation_partners_are_disjoint_from_training_partners_and_sub_policies(layout, population):
    # "sp" only here: "sp+biased" and "hsp-w1" read the pool and are tested below.
    train = set(experiment.train_partner_paths("sp", layout).values())
    subs = set(experiment.population_paths(population, layout).values())
    for group, paths in experiment.heldout_partner_groups(population, layout).items():
        held_out = set(paths.values())
        assert not held_out & train, f"{group} overlaps the training partners"
        assert not held_out & subs, f"{group} overlaps the ensemble's sub-policies"


def test_sub_policies_never_train_against_themselves():
    train = set(experiment.train_partner_paths("sp", "unident_s").values())
    for population in experiment.POPULATIONS:
        if population not in _POOL_POPULATIONS:
            assert not train & set(experiment.population_paths(population, "unident_s").values())


def test_biased_training_partners_exclude_the_benchmark_set():
    import os

    from adhoc_teamwork.population.zsceval_pool import benchmark_partner_paths, default_pool_root

    if not os.path.isdir(os.path.join(default_pool_root(), "unident_s")):
        pytest.skip("no local policy pool")
    train = set(experiment.train_partner_paths("sp+biased", "unident_s").values())
    assert len(train) > len(experiment.train_partner_paths("sp", "unident_s"))
    assert not train & set(benchmark_partner_paths("unident_s").values())


def test_specialist_population_never_uses_benchmark_experiments():
    import os
    import re

    from adhoc_teamwork.population.zsceval_pool import _benchmark_exps, default_pool_root

    if not os.path.isdir(os.path.join(default_pool_root(), "unident_s")):
        pytest.skip("no local policy pool")
    names = experiment.population_paths("hsp-w1", "unident_s")
    assert len(names) == 10
    exps = {int(re.match(r"hsp(\d+)_w1", n).group(1)) for n in names}
    assert len(exps) == 10
    assert not exps & set(_benchmark_exps("unident_s"))
    assert all(p.endswith("_final_w1_actor.pt") for p in names.values())


@pytest.mark.parametrize("layout", ["unident_s", "random0_medium", "random0", "random1", "random3", "small_corridor"])
def test_traj_full10_is_a_whole_population(layout):
    """Guards against the pool's swapped traj-S1-s10 / traj-S1-s15 folder labels."""
    import os

    from adhoc_teamwork.population.zsceval_pool import default_pool_root

    root = default_pool_root()
    if not os.path.isdir(os.path.join(root, layout)):
        pytest.skip("no local policy pool")
    paths = experiment.population_paths("traj-full10", layout)
    assert len(paths) == 10
    assert all(os.path.exists(os.path.join(root, p)) for p in paths.values())
    folder = os.path.dirname(os.path.join(root, next(iter(paths.values()))))
    assert len([f for f in os.listdir(folder) if f.endswith("_final_actor.pt")]) == 10


def test_baselines_include_cole_and_e3t_and_exclude_held_out_partners():
    from adhoc_teamwork.evaluation.aggregate import method_family

    paths = experiment.baseline_paths("unident_s")
    families = {method_family(name) for name in paths}
    assert {"mappo", "traj_s36", "hsp_s36", "fcp_s36", "mep_s36", "cole_s75", "e3t"} == families
    held_out = set(experiment.heldout_partner_groups("traj-full10", "unident_s")["biased"].values())
    assert not held_out & set(paths.values())


def test_adding_baselines_did_not_change_the_stage2_ensemble():
    members = experiment.population_paths("s2-mix", "unident_s")
    assert len(members) == 12
    assert all("/cole/" not in p and "/e3t/" not in p for p in members.values())
    baselines = experiment.baseline_paths("unident_s")
    assert {"cole_s75_1", "e3t_1", "mappo_sp1_final", "hsp_s36_5"} <= set(baselines)
    assert len(baselines) == 35


def test_all_baseline_checkpoints_load_with_weights_only():
    """Covers COLE and E3T, whose architecture must match what the wrapper expects."""
    import os

    import torch

    from adhoc_teamwork.policies.zsceval_policy import load_zsceval_policy
    from adhoc_teamwork.population.zsceval_pool import default_pool_root

    root = default_pool_root()
    if not os.path.isdir(os.path.join(root, "unident_s")):
        pytest.skip("no local policy pool")
    for name, path in experiment.baseline_paths("unident_s").items():
        policy = load_zsceval_policy(root, "unident_s", path)
        probs, _ = policy.action_probs(torch.zeros(1, policy.obs_dim), policy.initial_state(1))
        assert torch.allclose(probs.sum(-1), torch.ones(1)), name


def test_new_specialists_share_reward_styles_with_the_pool_but_not_files():
    import os

    from adhoc_teamwork.population.zsceval_pool import default_pool_root

    if not os.path.isdir(os.path.join(default_pool_root(), "unident_s")):
        pytest.skip("no local policy pool")
    pool = experiment.population_paths("hsp-w1", "unident_s")
    new = experiment.population_paths("hsp-w1-new", "unident_s")
    assert list(pool) == list(new), "same names means the same 10 reward styles"
    root = experiment.new_policy_root()
    assert all(os.path.isabs(p) and p.startswith(root) for p in new.values())
    assert not set(pool.values()) & set(new.values())
    assert all(os.path.basename(new[n]) == os.path.basename(pool[n]) for n in pool)


def test_extraction_only_uses_finished_runs(tmp_path):
    from adhoc_teamwork.scripts.extract_new_policies import collect

    for run, step in (("run1", 5000000), ("run2", 9960000)):
        d = tmp_path / run
        d.mkdir()
        (d / f"actor_periodic_{step}.pt").write_text(run)
    unfinished = tmp_path / "other"
    unfinished.mkdir()
    (unfinished / "actor_periodic_5000000.pt").write_text("partial")

    out = tmp_path / "out"
    sources = {"done.pt": str(tmp_path / "run*" / "actor_periodic_*.pt"), "partial.pt": str(unfinished / "actor_periodic_*.pt")}
    assert collect(sources, str(out)) == 1
    assert (out / "done.pt").read_text() == "run2", "must take the finished run, not the preempted one"
    assert not (out / "partial.pt").exists()


def _need_pool(layout="unident_s"):
    import os

    from adhoc_teamwork.population.zsceval_pool import default_pool_root

    if not os.path.isdir(os.path.join(default_pool_root(), layout)):
        pytest.skip("no local policy pool")


@pytest.mark.parametrize("layout", ["unident_s", "random0_medium", "random3"])
@pytest.mark.parametrize("population", ["hsp-w1", "s2-mix", "traj-full10"] + list(experiment.MIXED_POPULATIONS))
def test_no_training_set_overlaps_any_evaluation_group(layout, population):
    """Covers every partner set, including the generator-held-out ones, and the
    files they point to exist."""
    import os

    from adhoc_teamwork.population.zsceval_pool import default_pool_root

    _need_pool(layout)
    root = default_pool_root()
    subs = set(experiment.population_paths(population, layout).values())
    for partner_set in experiment.TRAIN_PARTNER_SETS:
        train = set(experiment.train_partner_paths(partner_set, layout).values())
        assert all(os.path.exists(os.path.join(root, p)) for p in train), partner_set
        assert not train & subs, f"{partner_set} trains sub-policies against themselves"
        for group, paths in experiment.heldout_partner_groups(population, layout).items():
            assert not set(paths.values()) & train, f"{group} overlaps {partner_set}"
            assert not set(paths.values()) & subs, f"{group} overlaps the sub-policies"
            assert all(os.path.exists(os.path.join(root, p)) for p in paths.values()), group


def test_generator_held_out_sets_leave_out_exactly_one_generator():
    _need_pool()
    everything = experiment.train_partner_groups("all-gen", "unident_s")
    assert set(everything) == set(experiment.TRAINING_GENERATORS)
    for generator in experiment.TRAINING_GENERATORS:
        groups = experiment.train_partner_groups(f"no-{generator}", "unident_s")
        assert set(groups) == set(experiment.TRAINING_GENERATORS) - {generator}
        assert experiment.balanced_over_generators(f"no-{generator}")
    assert not experiment.balanced_over_generators("sp+biased"), "keeps the original runs reproducible"
    assert sum(len(g) for g in everything.values()) == len(experiment.train_partner_paths("all-gen", "unident_s")), "no name collisions"


def test_partner_groups_keep_their_original_order_and_unique_names():
    """Episode seeds follow this order, so existing results stay comparable."""
    groups = experiment.all_partner_groups("unident_s")
    assert list(groups)[:3] == ["biased", "mep", "self_play"]
    names = [n for g in groups.values() for n in g]
    assert len(names) == len(set(names))
    assert list(groups["self_play"])[:5] == [f"sp{s}_final" for s in range(1, 6)]
    assert len(groups["cole"]) == 10 and len(groups["e3t"]) == 5


def test_stage2_populations_skip_groups_their_members_trained_with():
    for population in ["s2-mix"] + list(experiment.MIXED_POPULATIONS):
        groups = experiment.heldout_partner_groups(population, "unident_s")
        assert "self_play" not in groups and "traj" not in groups
        assert "mep" in groups, "stage-2 agents trained with mep-S1-s15, not the s10 population"
    assert "traj" not in experiment.heldout_partner_groups("traj-full10", "unident_s")
    assert set(experiment.heldout_partner_groups("hsp-w1", "unident_s")) == set(experiment.all_partner_groups("unident_s"))


def test_mixed_populations_have_the_stated_ratios():
    import re

    _need_pool()
    for population, (seeds, specialists, traj_members) in experiment.MIXED_POPULATIONS.items():
        names = list(experiment.population_paths(population, "unident_s"))
        assert len(names) == 12
        assert sum(n.endswith("_w1") for n in names) == specialists
        assert sum(n.startswith("traj") and n.endswith("_final") for n in names) == traj_members
        assert sum(bool(re.match(r"^(traj|hsp|fcp|mep)_s36_\d+$", n)) for n in names) == 4 * seeds
