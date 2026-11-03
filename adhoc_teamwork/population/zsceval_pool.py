"""Named policy sets from a ZSC-Eval ``policy_pool`` directory.

The proposal (Section 4) suggests building the ensemble from "existing ZSC
baselines such as TrajDi or FCP" and evaluating against "randomly generated
individual biased agents". ZSC-Eval's pool provides both, plus trained
versions of the baselines the proposal compares against. This module only
maps *names* to checkpoints; which set plays which role in an experiment is
decided in ``experiment.py``.

The pool itself is not in this repository. Get it with::

    git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool

and point ``--pool-root`` or ``$ZSC_POLICY_POOL`` at it.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import torch

from adhoc_teamwork.policies.zsceval_policy import load_zsceval_policy
from adhoc_teamwork.population.policy_pool import PolicyPool

DEFAULT_POOL_ROOT = os.path.expanduser("~/ZSC-Eval/zsceval/policy_pool")


def default_pool_root() -> str:
    return os.environ.get("ZSC_POLICY_POOL", DEFAULT_POOL_ROOT)


def _benchmark_exps(layout: str) -> List[int]:
    from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS

    return list(LAYOUTS_EXPS[layout])


def self_play_paths(layout: str, seeds: Iterable[int], tags: Iterable[str] = ("final",)) -> Dict[str, str]:
    """Independently-trained MAPPO self-play agents (ZSC-Eval's FCP stage 1)."""
    return {f"sp{s}_{t}": f"{layout}/fcp/s1/sp/sp{s}_{t}_actor.pt" for s in seeds for t in tags}


def stage1_population_paths(layout: str, algo: str, size: int, members: Iterable[int], tags: Iterable[str] = ("final",)) -> Dict[str, str]:
    """Members of a diversity-trained stage-1 population (``algo`` in traj, mep)."""
    return {f"{algo}{m}_{t}": f"{layout}/{algo}/s1/{algo}-S1-s{size}/{algo}{m}_{t}_actor.pt" for m in members for t in tags}


def benchmark_partner_paths(layout: str, tags: Iterable[str] = ("mid", "final")) -> Dict[str, str]:
    """ZSC-Eval's held-out evaluation partners: the biased agents its
    Best-Response-Diversity selection picked for this layout, under the same
    ``bias{i}_{tag}`` names its published benchmark results use.
    """
    return {f"bias{i + 1}_{t}": f"{layout}/hsp/s1/hsp/hsp{exp}_{t}_w0_actor.pt" for i, exp in enumerate(_benchmark_exps(layout)) for t in tags}


def non_benchmark_exps(layout: str, pool_root: str = None) -> List[int]:
    """HSP experiments the benchmark selection did *not* pick, so usable without
    touching the held-out partners."""
    root = pool_root or default_pool_root()
    held_out = set(_benchmark_exps(layout))
    directory = os.path.join(root, layout, "hsp", "s1", "hsp")
    available = sorted(int(f.split("_")[0][3:]) for f in os.listdir(directory) if f.endswith("_final_w0_actor.pt"))
    return [e for e in available if e not in held_out]


def non_benchmark_biased_paths(layout: str, count: int, tags: Iterable[str] = ("final",), pool_root: str = None) -> Dict[str, str]:
    """Biased agents (w0) outside the benchmark set. Safe to train against, but
    drawn from the same generator as it, so they do not test generalization
    across generators.
    """
    exps = non_benchmark_exps(layout, pool_root)[:count]
    return {f"hsp{e}_{t}": f"{layout}/hsp/s1/hsp/hsp{e}_{t}_w0_actor.pt" for e in exps for t in tags}


def biased_counterpart_paths(layout: str, count: int, pool_root: str = None) -> Dict[str, str]:
    """The task-reward agents (w1) co-trained with non-benchmark biased agents:
    specialists, each adapted to one partner style. Evenly spaced across the
    available experiments rather than chosen using any evaluation result.
    """
    exps = non_benchmark_exps(layout, pool_root)
    picks = [exps[i] for i in np.linspace(0, len(exps) - 1, count).round().astype(int)]
    return {f"hsp{e}_w1": f"{layout}/hsp/s1/hsp/hsp{e}_final_w1_actor.pt" for e in picks}


def stage2_baseline_paths(layout: str, algo: str, population_size: int, seeds: Iterable[int]) -> Dict[str, str]:
    """Trained best-response baselines (``algo`` in traj, hsp, fcp, mep, cole)."""
    return {f"{algo}_s{population_size}_{s}": f"{layout}/{algo}/s2/{algo}-S2-s{population_size}/{s}.pt" for s in seeds}


def load_pool(paths: Dict[str, str], layout: str, pool_root: str = None, device: torch.device = torch.device("cpu")) -> PolicyPool:
    root = pool_root or default_pool_root()
    pool = PolicyPool()
    for name, rel_path in paths.items():
        pool.add(name, load_zsceval_policy(root, layout, rel_path, device=device, name=name))
    return pool
