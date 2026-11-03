"""The train/evaluation partner split is what makes "unseen partner" mean
anything (Section 6, Section 9), so it is tested rather than assumed."""

import pytest

pytest.importorskip("zsceval")

from adhoc_teamwork import experiment  # noqa: E402


@pytest.mark.parametrize("layout", ["unident_s", "random0_medium"])
@pytest.mark.parametrize("population", [p for p in experiment.POPULATIONS if p != "hsp-w1"])
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
        if population != "hsp-w1":
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
