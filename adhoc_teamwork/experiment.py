"""Which policies play which role in a real-Overcooked experiment.

The important constraint (Section 6: "held-out biased partners"; Section 9:
a method must not "simply approximate the expected partner behaviors in the
testing set") is that nothing the weighting network trains against may be in
the evaluation set. ``tests/test_experiment.py`` checks this.

What ZSC-Eval's own stage-2 agents trained with matters too, both for which
partner groups an ensemble of them can be evaluated on and for which baselines
a group is held out from. From ZSC-Eval's ``scripts/prep/gen_S2_yml.py``,
``gen_hsp_S2_ymls.py`` and ``scripts/overcooked/shell/train_*_stage_2.sh``, the
36-member populations are: FCP, self-play seeds 1-15; MEP, ``mep-S1-s15``;
TrajeDi, ``traj-S1-s15``; HSP, biased agents outside the benchmark set plus
``mep-S1-s15``.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from adhoc_teamwork.population.zsceval_pool import (
    benchmark_partner_paths,
    biased_counterpart_paths,
    e3t_baseline_paths,
    non_benchmark_biased_paths,
    self_play_paths,
    stage1_population_paths,
    stage2_baseline_paths,
)

# Self-play seeds are split between roles so that no seed plays two of them.
ENSEMBLE_SP_SEEDS = range(1, 9)
TRAIN_SP_SEEDS = range(9, 16)
HELDOUT_SP_SEEDS = range(1, 9)  # only held out when the ensemble is not built from self-play

# Mixtures of agent types, as (seeds of each stage-2 algorithm, specialists, TrajeDi
# members). Each has 12 members, like s2-mix, so only the ratio changes.
MIXED_POPULATIONS = {"mix-g8s4": (2, 4, 0), "mix-g4s8": (1, 8, 0), "mix-g4s4t4": (1, 4, 4)}
POPULATIONS = ("traj-full10", "traj-s10", "mep-s10", "sp8", "s2-mix", "hsp-w1", "traj-new10", "hsp-w1-new") + tuple(MIXED_POPULATIONS)
# Generators of training partners, and the sets built from them: "all-gen" uses
# all four, "no-<g>" all but one (to test on that one as an unseen generator).
TRAINING_GENERATORS = ("sp", "biased", "mep", "traj")
TRAIN_PARTNER_SETS = ("sp", "sp+biased", "all-gen") + tuple(f"no-{g}" for g in TRAINING_GENERATORS)
BASELINE_ALGOS = ("traj", "hsp", "fcp", "mep")
BASELINE_SEEDS = range(1, 6)
# The folder holding the whole 10-member TrajeDi population; see population_paths.
TRAJ_FULL10_FOLDER_SIZE = {"random0": 10}
TRAJ_FULL10_DEFAULT_FOLDER_SIZE = 15
_TAGS = ("mid", "final")


def new_policy_root() -> str:
    """Where ``scripts/extract_new_policies.py`` puts policies trained in this project."""
    default = os.path.join(os.environ.get("ADHOC_SCRATCH", os.path.expanduser("~/adhoc_teamwork")), "new_policies")
    return os.environ.get("ZSC_NEW_POLICIES", default)


def traj_full10_folder_size(layout: str) -> int:
    return TRAJ_FULL10_FOLDER_SIZE.get(layout, TRAJ_FULL10_DEFAULT_FOLDER_SIZE)


def traj_other_folder_size(layout: str) -> int:
    """The folder of the 15-member TrajeDi population (labels swapped; see population_paths)."""
    return 10 if traj_full10_folder_size(layout) == 15 else 15


def population_paths(population: str, layout: str, pool_root: str = None) -> Dict[str, str]:
    """The ensemble's sub-policies (Section 4: "use existing ZSC baselines
    such as TrajDi or FCP" to build the population).

    * ``traj-full10``, ``traj-s10``, ``mep-s10``, ``sp8``: self-play-style agents,
      diverse with each other but never trained with anyone outside their own
      population. On every layout we use but ``random0``, the pool's ``traj-S1-s10``
      folder actually holds 15 members and ``traj-S1-s15`` holds 10 (the MEP folders
      are correct), so the labels appear swapped. ``traj-full10`` is the whole
      10-member population, whichever folder holds it. ``traj-s10`` always reads the
      folder labeled s10, kept so that earlier results stay reproducible.
    * ``s2-mix``: stage-2 agents of several algorithms, each already trained
      to be robust. The baselines themselves are its members.
    * ``hsp-w1``: specialists, each trained alongside one biased partner.
    * ``traj-new10``, ``hsp-w1-new``: the same kinds of policy as ``traj-full10`` and
      ``hsp-w1``, trained here with new random seeds. ``hsp-w1-new`` uses the same 10
      biased reward styles as ``hsp-w1``. Their absolute paths bypass the pool root.
    * ``mix-*`` (see ``MIXED_POPULATIONS``): stage-2 generalists, specialists and
      TrajeDi members in different ratios, 12 members in all.
    """
    if population in MIXED_POPULATIONS:
        seeds, specialists, traj_members = MIXED_POPULATIONS[population]
        paths = {}
        for algo in BASELINE_ALGOS:
            paths.update(stage2_baseline_paths(layout, algo, 36, range(1, seeds + 1)))
        if specialists:
            paths.update(biased_counterpart_paths(layout, specialists, pool_root=pool_root))
        if traj_members:
            members = np.linspace(1, 10, traj_members).round().astype(int).tolist()
            paths.update(stage1_population_paths(layout, "traj", traj_full10_folder_size(layout), members))
        return paths
    if population == "traj-new10":
        return {f"traj{m}_final": os.path.join(new_policy_root(), layout, "traj", f"traj{m}_final_actor.pt") for m in range(1, 11)}
    if population == "hsp-w1-new":
        return {name: os.path.join(new_policy_root(), layout, "hsp", os.path.basename(path)) for name, path in biased_counterpart_paths(layout, 10, pool_root=pool_root).items()}
    if population == "s2-mix":
        paths = {}
        for algo in BASELINE_ALGOS:
            paths.update(stage2_baseline_paths(layout, algo, 36, range(1, 4)))
        return paths
    if population == "hsp-w1":
        return biased_counterpart_paths(layout, 10, pool_root=pool_root)
    if population == "traj-full10":
        return stage1_population_paths(layout, "traj", traj_full10_folder_size(layout), range(1, 11))  # see the note above
    if population == "traj-s10":
        return stage1_population_paths(layout, "traj", 10, range(1, 11))
    if population == "mep-s10":
        return stage1_population_paths(layout, "mep", 10, range(1, 11))
    if population == "sp8":
        return self_play_paths(layout, ENSEMBLE_SP_SEEDS)
    raise ValueError(f"unknown population {population!r}; choose from {POPULATIONS}")


def has_stage2_members(population: str) -> bool:
    return population == "s2-mix" or population in MIXED_POPULATIONS


def has_traj_members(population: str) -> bool:
    return population.startswith("traj-") or (population in MIXED_POPULATIONS and MIXED_POPULATIONS[population][2] > 0)


def _stage1_folder_paths(layout: str, algo: str, folder_size: int, members) -> Dict[str, str]:
    """Like stage1_population_paths, with the folder in the name, so members of
    different populations do not collide."""
    paths = stage1_population_paths(layout, algo, folder_size, members, tags=_TAGS)
    return {f"{algo}-s{folder_size}-{name[len(algo):]}": path for name, path in paths.items()}


def training_generator_paths(layout: str, pool_root: str = None) -> Dict[str, Dict[str, str]]:
    """Training partners from each generator, all at partial and full skill.
    None is in any evaluation group: self-play seeds 9-15 (held out: 1-8), biased
    agents outside the benchmark set, and MEP and TrajeDi populations other than
    the ones evaluated on (``mep-S1-s10`` and the 10-member TrajeDi population).
    """
    traj_other = traj_other_folder_size(layout)
    return {
        "sp": self_play_paths(layout, TRAIN_SP_SEEDS, tags=_TAGS),
        "biased": non_benchmark_biased_paths(layout, count=20, tags=_TAGS, pool_root=pool_root),
        "mep": {**_stage1_folder_paths(layout, "mep", 15, range(1, 16)), **_stage1_folder_paths(layout, "mep", 5, range(1, 6))},
        "traj": {**_stage1_folder_paths(layout, "traj", traj_other, range(1, 16)), **_stage1_folder_paths(layout, "traj", 5, range(1, 6))},
    }


def train_partner_groups(partner_set: str, layout: str, pool_root: str = None) -> Dict[str, Dict[str, str]]:
    """What the encoder and weighting network train against, by generator.

    * ``"sp"``: self-play agents, FCP-style (Strouse et al., 2021) at partial
      and full skill. Biased agents are never seen, so evaluating on them
      tests generalization to a different generator of partners.
    * ``"sp+biased"``: also includes biased agents *outside* the benchmark set.
    * ``"all-gen"``: self-play, biased, MEP and TrajeDi agents.
    * ``"no-<generator>"``: all of those but one, which is then an unseen generator.
    """
    if partner_set not in TRAIN_PARTNER_SETS:
        raise ValueError(f"unknown partner set {partner_set!r}; choose from {TRAIN_PARTNER_SETS}")
    if partner_set == "sp":
        return {"sp": self_play_paths(layout, TRAIN_SP_SEEDS, tags=_TAGS)}
    if partner_set == "sp+biased":
        return {"sp": self_play_paths(layout, TRAIN_SP_SEEDS, tags=_TAGS), "biased": non_benchmark_biased_paths(layout, count=20, tags=_TAGS, pool_root=pool_root)}
    generators = training_generator_paths(layout, pool_root)
    left_out = partner_set[len("no-"):] if partner_set.startswith("no-") else None
    return {g: paths for g, paths in generators.items() if g != left_out}


def balanced_over_generators(partner_set: str) -> bool:
    """Whether training samples a generator first and then one of its agents.
    The two original sets sample agents uniformly, as their results were made
    that way; with more generators, that would let the largest one dominate.
    """
    return partner_set not in ("sp", "sp+biased")


def train_partner_paths(partner_set: str, layout: str, pool_root: str = None) -> Dict[str, str]:
    """All of a set's training partners, in one dict."""
    paths = {}
    for group in train_partner_groups(partner_set, layout, pool_root).values():
        paths.update(group)
    return paths


def all_partner_groups(layout: str) -> Dict[str, Dict[str, str]]:
    """Every evaluation group, in a fixed order that also fixes each partner's
    episode seeds (``scripts/benchmark_overcooked.py``). New groups go at the end.

    ``biased`` is the primary comparison: the same partners as ZSC-Eval's
    published benchmark. The rest come from other generators: a MEP population
    and self-play agents (unseen by the HSP and MEP baselines), a TrajeDi
    population, COLE agents from smaller populations than the COLE baseline, and
    E3T agents, which are also the E3T baseline.
    """
    cole = {**stage2_baseline_paths(layout, "cole", 25, BASELINE_SEEDS), **stage2_baseline_paths(layout, "cole", 50, BASELINE_SEEDS)}
    return {
        "biased": benchmark_partner_paths(layout),
        "mep": stage1_population_paths(layout, "mep", 10, range(1, 11)),
        "self_play": self_play_paths(layout, HELDOUT_SP_SEEDS),
        "traj": stage1_population_paths(layout, "traj", traj_full10_folder_size(layout), range(1, 11)),
        "cole": cole,
        "e3t": e3t_baseline_paths(layout, BASELINE_SEEDS),
    }


# Baselines a group is not held out from, because they trained with its agents
# or are them. TrajeDi's are included because, with the swapped folder labels,
# which population its stage-2 agents trained with is uncertain.
GROUP_SEEN_BY_BASELINES: Dict[str, Tuple[str, ...]] = {"self_play": ("mappo", "fcp_s36"), "traj": ("traj_s36",), "e3t": ("e3t",)}


def heldout_partner_groups(population: str, layout: str) -> Dict[str, Dict[str, str]]:
    """Evaluation partners for one population, by group: all groups but those
    that overlap its sub-policies or what they trained with.
    """
    excluded = set()
    if has_stage2_members(population):
        excluded |= {"self_play", "traj"}  # FCP trained with self-play 1-15; see GROUP_SEEN_BY_BASELINES
    if has_traj_members(population):
        excluded.add("traj")
    if population == "mep-s10":
        excluded.add("mep")
    if population == "sp8":
        excluded.add("self_play")
    return {group: paths for group, paths in all_partner_groups(layout).items() if group not in excluded}


def baseline_paths(layout: str) -> Dict[str, str]:
    """Single-policy baselines from Section 4: MAPPO (self-play), TrajeDi and
    HSP, plus FCP, MEP, COLE and E3T. Stage-2 agents use their largest trained
    population (36, or 75 for COLE).
    """
    paths = {f"mappo_{name}": path for name, path in self_play_paths(layout, BASELINE_SEEDS).items()}
    for algo in BASELINE_ALGOS:
        paths.update(stage2_baseline_paths(layout, algo, 36, BASELINE_SEEDS))
    paths.update(stage2_baseline_paths(layout, "cole", 75, BASELINE_SEEDS))
    paths.update(e3t_baseline_paths(layout, BASELINE_SEEDS))
    return paths
