#!/usr/bin/env python
"""Train the partner trajectory encoder + adaptive weighting network on top
of a subset ("held-in") of the trained population (Section 7 deliverables:
"a partner trajectory encoder" and "an adaptive ensemble weighting model").

Usage (gridworld, population from scripts/train_population.py):
    python -m adhoc_teamwork.scripts.train_adaptive_ensemble \
        --population results/population --held-in policy_0 policy_1 \
        --out results/ensemble --n-iterations 150

Usage (real Overcooked, population named in experiment.py; see slurm/):
    python -m adhoc_teamwork.scripts.train_adaptive_ensemble --env overcooked --layout unident_s \
        --population traj-s10 --train-partners sp --out <dir> --n-iterations 1000 \
        --episodes-per-iteration 64 --partners-per-iteration 8 --advantage actor_critic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.nn_agent import NNAgent
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.population.policy_pool import PolicyPool
from adhoc_teamwork.training.train_adaptive_ensemble import ACTION_RULES, EnsembleTrainConfig, train_adaptive_ensemble

# The weighting network is trained against a *training* distribution of
# partners (here: moderately-noisy scripted roles) that is deliberately
# narrower than what it is evaluated against (see scripts/evaluate.py),
# so the final numbers measure generalization to unseen partners rather
# than memorization of the exact training distribution.
TRAIN_PARTNER_ROLES = [("onion_runner", 0.0), ("onion_runner", 0.1), ("pot_tender", 0.0), ("pot_tender", 0.1)]


def make_partner_sampler():
    def sample():
        role, noise = random.choice(TRAIN_PARTNER_ROLES)
        return ScriptedBiasedAgent(role, noise=noise)

    return sample


def make_overcooked_setup(args, device):
    """(env constructor, population, partner sampler) on the real env."""
    from adhoc_teamwork import experiment
    from adhoc_teamwork.envs.zsceval_adapter import SOUP_DELIVERY_REWARD, make_overcooked_env
    from adhoc_teamwork.population.zsceval_pool import load_pool

    pool = load_pool(experiment.population_paths(args.population, args.layout, args.pool_root), args.layout, args.pool_root, device)
    generators = experiment.train_partner_groups(args.train_partners, args.layout, args.pool_root)
    partner_pool = load_pool(experiment.train_partner_paths(args.train_partners, args.layout, args.pool_root), args.layout, args.pool_root, device)
    agents = {name: NNAgent(partner_pool[name], device) for name in partner_pool.names()}
    rng = random.Random(args.seed)
    if experiment.balanced_over_generators(args.train_partners):
        by_generator = [[agents[name] for name in paths] for paths in generators.values()]
        sampler = lambda: rng.choice(rng.choice(by_generator))  # noqa: E731
    else:
        partners = list(agents.values())
        sampler = lambda: rng.choice(partners)  # noqa: E731
    return (lambda: make_overcooked_env(args.layout, horizon=args.horizon)), pool, sampler, SOUP_DELIVERY_REWARD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["gridworld", "overcooked"], default="gridworld")
    parser.add_argument("--layout", type=str, default="unident_s", help="Overcooked layout (--env overcooked)")
    parser.add_argument("--pool-root", type=str, default=None, help="ZSC-Eval policy pool (--env overcooked)")
    parser.add_argument("--train-partners", type=str, default="sp", help="partner set from experiment.py (--env overcooked)")
    parser.add_argument("--population", type=str, default="results/population", help="directory (gridworld) or population name (overcooked)")
    parser.add_argument("--held-in", type=str, nargs="+", default=None, help="policy names to use as sub-policies (default: all)")
    parser.add_argument("--out", type=str, default="results/ensemble")
    parser.add_argument("--horizon", type=int, default=None, help="default: 150 on gridworld, 400 on overcooked")
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--episodes-per-iteration", type=int, default=4)
    parser.add_argument("--partners-per-iteration", type=int, default=None, help="default: a new partner per episode")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--encoder-hidden", type=int, default=32)
    parser.add_argument("--weighting-hidden", type=int, default=32)
    parser.add_argument(
        "--encoder-ablation",
        choices=["none", "no_partner_action", "memoryless"],
        default="none",
        help="no_partner_action: hide the partner's actions from the encoder. memoryless: also remove its memory, so weights depend only on the current state.",
    )
    parser.add_argument("--advantage", choices=["reinforce", "actor_critic"], default="reinforce")
    parser.add_argument(
        "--action-rule",
        choices=ACTION_RULES,
        default="sample",
        help="how the ensemble acts while training, matching an evaluation method: sample (adaptive_ensemble), argmax "
        "(adaptive_ensemble_argmax), select (policy_selection) or select_argmax (policy_selection_argmax)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-entropy-coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    if args.env == "overcooked":
        args.horizon = args.horizon or 400
        env, pool, partner_sampler, reward_scale = make_overcooked_setup(args, device)
    else:
        args.horizon = args.horizon or 150
        pool = PolicyPool.load(args.population)
        env, partner_sampler, reward_scale = OvercookedGridworld(horizon=args.horizon), make_partner_sampler(), 1.0

    held_in = args.held_in or pool.names()
    sub_policies = [pool[name] for name in held_in]

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    encoder_kwargs = {"use_partner_actions": args.encoder_ablation == "none", "use_memory": args.encoder_ablation != "memoryless"}
    t0 = time.time()
    encoder, weighting_net, stats = train_adaptive_ensemble(
        env,
        sub_policies,
        partner_sampler,
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        weighting_hidden=args.weighting_hidden,
        n_iterations=args.n_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        partners_per_iteration=args.partners_per_iteration,
        cfg=EnsembleTrainConfig(lr=args.lr, weight_entropy_coef=args.weight_entropy_coef, advantage=args.advantage, reward_scale=reward_scale, action_rule=args.action_rule),
        device=device,
        seed=args.seed,
        verbose=not args.quiet,
        log_every=max(1, args.n_iterations // 100),
        checkpoint_path=f"{args.out}/train_state.pt",
        encoder_kwargs=encoder_kwargs,
    )
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "weighting_state_dict": weighting_net.state_dict(),
            "held_in": held_in,
            "action_rule": args.action_rule,
            "embed_dim": args.embed_dim,
            "encoder_hidden": encoder.hidden_size,
            **encoder_kwargs,
            "weighting_hidden": args.weighting_hidden,
            "obs_dim": encoder.obs_dim,
            "num_actions": encoder.num_actions,
        },
        f"{args.out}/ensemble.pt",
    )
    with open(f"{args.out}/train_returns.txt", "w") as f:
        f.write("\n".join(str(r) for r in stats.returns))
    with open(f"{args.out}/train_stats.json", "w") as f:
        json.dump(vars(stats), f)
    print(f"Trained adaptive ensemble on sub-policies {held_in} in {time.time() - t0:.1f}s")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
