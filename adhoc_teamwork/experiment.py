"""Which policies play which role in a real-Overcooked experiment.

The important constraint (Section 6: "held-out biased partners"; Section 9:
a method must not "simply approximate the expected partner behaviors in the
testing set") is that nothing the weighting network trains against may be in
the evaluation set. ``tests/test_experiment.py`` checks this.
"""

from __future__ import annotations

import os
from typing import Dict

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
HELDOUT_SP_SEEDS = range(1, 6)  # only held out when the ensemble is not built from self-play

POPULATIONS = ("traj-full10", "traj-s10", "mep-s10", "sp8", "s2-mix", "hsp-w1", "traj-new10", "hsp-w1-new")
TRAIN_PARTNER_SETS = ("sp", "sp+biased")
BASELINE_ALGOS = ("traj", "hsp", "fcp", "mep")
BASELINE_SEEDS = range(1, 6)
# The folder holding the whole 10-member TrajeDi population; see population_paths.
TRAJ_FULL10_FOLDER_SIZE = {"random0": 10}
TRAJ_FULL10_DEFAULT_FOLDER_SIZE = 15


def new_policy_root() -> str:
    """Where ``scripts/extract_new_policies.py`` puts policies trained in this project."""
    default = os.path.join(os.environ.get("ADHOC_SCRATCH", os.path.expanduser("~/adhoc_teamwork")), "new_policies")
    return os.environ.get("ZSC_NEW_POLICIES", default)


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
    """
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
        folder_size = TRAJ_FULL10_FOLDER_SIZE.get(layout, TRAJ_FULL10_DEFAULT_FOLDER_SIZE)
        return stage1_population_paths(layout, "traj", folder_size, range(1, 11))  # see the note above
    if population == "traj-s10":
        return stage1_population_paths(layout, "traj", 10, range(1, 11))
    if population == "mep-s10":
        return stage1_population_paths(layout, "mep", 10, range(1, 11))
    if population == "sp8":
        return self_play_paths(layout, ENSEMBLE_SP_SEEDS)
    raise ValueError(f"unknown population {population!r}; choose from {POPULATIONS}")


def train_partner_paths(partner_set: str, layout: str, pool_root: str = None) -> Dict[str, str]:
    """What the encoder and weighting network train against.

    * ``"sp"``: self-play agents, FCP-style (Strouse et al., 2021) at partial
      and full skill. Biased agents are never seen, so evaluating on them
      tests generalization to a different generator of partners.
    * ``"sp+biased"``: also includes biased agents *outside* the benchmark set.
    """
    paths = self_play_paths(layout, TRAIN_SP_SEEDS, tags=("mid", "final"))
    if partner_set == "sp":
        return paths
    if partner_set == "sp+biased":
        paths.update(non_benchmark_biased_paths(layout, count=20, tags=("mid", "final"), pool_root=pool_root))
        return paths
    raise ValueError(f"unknown partner set {partner_set!r}; choose from {TRAIN_PARTNER_SETS}")


def heldout_partner_groups(population: str, layout: str) -> Dict[str, Dict[str, str]]:
    """Evaluation partners, by group. ``biased`` is the primary comparison: the
    same partners as ZSC-Eval's published benchmark. Groups that would overlap
    the ensemble's own sub-policies are left out.
    """
    groups = {"biased": benchmark_partner_paths(layout)}
    if population == "s2-mix":
        return groups  # stage-2 agents trained with the self-play and MEP agents
    if population != "mep-s10":
        groups["mep"] = stage1_population_paths(layout, "mep", 10, range(1, 11))
    if population != "sp8":
        groups["self_play"] = self_play_paths(layout, HELDOUT_SP_SEEDS)
    return groups


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
