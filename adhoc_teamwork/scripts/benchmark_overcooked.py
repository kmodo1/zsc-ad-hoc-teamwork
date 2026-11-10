#!/usr/bin/env python
"""Evaluate methods against the held-out partners on real Overcooked.

Three kinds of method, split so each is only evaluated as often as needed:

  --what population   each sub-policy alone and the uniform static ensemble,
                      each sampling and with argmax actions (once per layout
                      and population). The per-partner best
                      sub-policy shows what perfect per-partner selection from
                      this population would achieve. It is a reference rather
                      than a strict bound: switching within an episode could
                      in principle exceed it.
  --what baselines    MAPPO, TrajeDi, HSP, FCP, MEP, COLE, E3T (once per layout).
  --what ensemble     the trained adaptive ensemble, the policy-selection
                      baseline, and learned_static_ensemble (once per training
                      run), each sampling and with argmax actions. The latter
                      fixes the weights at their average over the *training*
                      partners, separating the benefit of adapting to the
                      partner from the benefit of simply favoring the better
                      sub-policies. Also records how the weights evolve over an
                      episode against each partner.

Baselines are evaluated on every partner group; anything built from a population
only on the groups held out from it (``experiment.heldout_partner_groups``).
Each partner's episode seeds depend only on its place in
``experiment.all_partner_groups``, so they are the same in every file.

Resumable; see evaluation/benchmark.py.
"""

from __future__ import annotations

import argparse
import functools
import json
import os

import numpy as np
import torch

from adhoc_teamwork import experiment
from adhoc_teamwork.baselines.policy_selection import PolicySelectionAgent
from adhoc_teamwork.baselines.static_ensemble import StaticEnsembleAgent, VoteEnsembleAgent
from adhoc_teamwork.ensemble.adaptive_ensemble_agent import AdaptiveEnsembleAgent
from adhoc_teamwork.ensemble.checkpoint import load_ensemble_checkpoint
from adhoc_teamwork.envs.vec_env import VecEnv
from adhoc_teamwork.envs.zsceval_adapter import make_overcooked_env
from adhoc_teamwork.evaluation.aggregate import method_family
from adhoc_teamwork.evaluation.benchmark import run_benchmark
from adhoc_teamwork.evaluation.rollout import run_episodes_batched
from adhoc_teamwork.policies.nn_agent import NNAgent
from adhoc_teamwork.population.zsceval_pool import default_pool_root, load_pool


def mean_weights_on_training_partners(args, make_adaptive, vec_env, device, load) -> np.ndarray:
    """The trained network's average weights over the partners it trained with.
    Cached, since a preempted job restarts from the top of the script.
    """
    run_dir = os.path.dirname(args.ensemble)
    cache = os.path.join(args.out, "learned_static_weights.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return np.array(json.load(f), dtype=np.float32)
    with open(os.path.join(run_dir, "config.json")) as f:
        train_partners = json.load(f)["train_partners"]
    pool = load(experiment.train_partner_paths(train_partners, args.layout, args.pool_root))
    traces = []
    for name in pool.names():
        for position in (0, 1):
            pair = [make_adaptive(), NNAgent(pool[name], device)]
            if position == 1:
                pair.reverse()
            result = run_episodes_batched(vec_env, pair, seed=args.seed, ego_index=position, record_weights=True)
            traces.append(result.ego_weights.mean(axis=(0, 1)))
    weights = np.mean(traces, axis=0).astype(np.float32)
    weights = weights / weights.sum()
    with open(cache, "w") as f:
        json.dump(weights.tolist(), f)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", required=True, choices=["population", "baselines", "ensemble"])
    parser.add_argument("--layout", required=True)
    parser.add_argument("--population", default="traj-s10", choices=experiment.POPULATIONS)
    parser.add_argument("--ensemble", default=None, help="checkpoint from train_adaptive_ensemble (for --what ensemble)")
    parser.add_argument("--argmax-actions", action="store_true", help="baselines take their most probable action, to compare like for like with the argmax ensemble")
    parser.add_argument("--baseline-families", nargs="+", default=None, help="evaluate only these baselines, e.g. cole_s75 e3t (default: all)")
    parser.add_argument("--partner-groups", nargs="+", default=None, help="evaluate only these partner groups, e.g. biased (default: all)")
    parser.add_argument("--pool-root", default=default_pool_root())
    parser.add_argument("--n-episodes", type=int, default=20, help="episodes per (method, partner, position)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    load = functools.partial(load_pool, layout=args.layout, pool_root=args.pool_root, device=device)

    all_groups = experiment.all_partner_groups(args.layout)
    partner_groups = all_groups if args.what == "baselines" else experiment.heldout_partner_groups(args.population, args.layout)
    if args.partner_groups:
        partner_groups = {group: partner_groups[group] for group in args.partner_groups}
    partner_paths = {}
    for group_paths in partner_groups.values():
        partner_paths.update(group_paths)
    partner_pool = load(partner_paths)
    partners = {name: functools.partial(NNAgent, partner_pool[name], device) for name in partner_pool.names()}

    methods = {}
    if args.what == "baselines":
        paths = experiment.baseline_paths(args.layout)
        if args.baseline_families:
            paths = {name: path for name, path in paths.items() if method_family(name) in args.baseline_families}
        baseline_pool = load(paths)
        methods = {name: functools.partial(NNAgent, baseline_pool[name], device, argmax_actions=args.argmax_actions) for name in baseline_pool.names()}
    else:
        population_pool = load(experiment.population_paths(args.population, args.layout, args.pool_root))
        sub_policies = [population_pool[name] for name in population_pool.names()]
        if args.what == "population":
            methods = {f"sub_{name}": functools.partial(NNAgent, population_pool[name], device) for name in population_pool.names()}
            methods.update({f"subargmax_{name}": functools.partial(NNAgent, population_pool[name], device, argmax_actions=True) for name in population_pool.names()})
            methods["static_ensemble"] = functools.partial(StaticEnsembleAgent, sub_policies, device=device)
            methods["static_ensemble_argmax"] = functools.partial(StaticEnsembleAgent, sub_policies, device=device, argmax_actions=True)
            methods["static_ensemble_vote"] = functools.partial(VoteEnsembleAgent, sub_policies, device=device)
        else:
            encoder, weighting_net, held_in = load_ensemble_checkpoint(args.ensemble, device=device)
            assert list(held_in) == population_pool.names(), f"checkpoint was trained on {held_in}"
            shared = (sub_policies, encoder, weighting_net)
            methods = {
                "adaptive_ensemble": functools.partial(AdaptiveEnsembleAgent, *shared, device=device),
                "adaptive_ensemble_argmax": functools.partial(AdaptiveEnsembleAgent, *shared, device=device, argmax_actions=True),
                "policy_selection": functools.partial(PolicySelectionAgent, *shared, device=device),
                "policy_selection_argmax": functools.partial(PolicySelectionAgent, *shared, device=device, argmax_actions=True),
            }

    vec_env = VecEnv(lambda: make_overcooked_env(args.layout), args.n_episodes)
    os.makedirs(args.out, exist_ok=True)
    if args.what == "ensemble":
        weights = mean_weights_on_training_partners(args, methods["adaptive_ensemble"], vec_env, device, load)
        print(f"[benchmark] learned static weights: {np.round(weights, 3).tolist()}", flush=True)
        methods["learned_static_ensemble"] = functools.partial(StaticEnsembleAgent, sub_policies, weights=weights, device=device)
        methods["learned_static_ensemble_argmax"] = functools.partial(StaticEnsembleAgent, sub_policies, weights=weights, device=device, argmax_actions=True)
    with open(os.path.join(args.out, "partner_groups.json"), "w") as f:
        json.dump({group: sorted(paths) for group, paths in partner_groups.items()}, f, indent=2)
    order = [name for group in all_groups.values() for name in group]
    partner_seeds = {name: args.seed + 1000 * order.index(name) for name in partners}
    run_benchmark(vec_env, methods, partners, os.path.join(args.out, "results.jsonl"), seed=args.seed, partner_seeds=partner_seeds)

    if args.what == "ensemble":
        weight_traces = {}
        for name, make_partner in partners.items():
            agent = methods["adaptive_ensemble"]()
            result = run_episodes_batched(vec_env, [agent, make_partner()], seed=args.seed, record_weights=True)
            weight_traces[name] = result.ego_weights.mean(axis=1)  # (T, K), averaged over episodes
        np.savez(os.path.join(args.out, "weight_traces.npz"), **weight_traces)


if __name__ == "__main__":
    main()
