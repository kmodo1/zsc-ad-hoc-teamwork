#!/usr/bin/env python
"""Train the partner trajectory encoder + adaptive weighting network on top
of a subset ("held-in") of the trained population (Section 7 deliverables:
"a partner trajectory encoder" and "an adaptive ensemble weighting model").

Usage:
    python -m adhoc_teamwork.scripts.train_adaptive_ensemble \
        --population results/population --held-in policy_0 policy_1 \
        --out results/ensemble --n-iterations 150
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.population.policy_pool import PolicyPool
from adhoc_teamwork.training.train_adaptive_ensemble import EnsembleTrainConfig, train_adaptive_ensemble

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=str, default="results/population")
    parser.add_argument("--held-in", type=str, nargs="+", default=None, help="policy names to use as sub-policies (default: all)")
    parser.add_argument("--out", type=str, default="results/ensemble")
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--episodes-per-iteration", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--encoder-hidden", type=int, default=32)
    parser.add_argument("--weighting-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pool = PolicyPool.load(args.population)
    held_in = args.held_in or pool.names()
    sub_policies = [pool[name] for name in held_in]

    env = OvercookedGridworld(horizon=args.horizon)
    t0 = time.time()
    encoder, weighting_net, stats = train_adaptive_ensemble(
        env,
        sub_policies,
        make_partner_sampler(),
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        weighting_hidden=args.weighting_hidden,
        n_iterations=args.n_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        cfg=EnsembleTrainConfig(),
        seed=args.seed,
        verbose=not args.quiet,
        log_every=max(1, args.n_iterations // 10),
    )
    import os

    os.makedirs(args.out, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "weighting_state_dict": weighting_net.state_dict(),
            "held_in": held_in,
            "embed_dim": args.embed_dim,
            "encoder_hidden": encoder.hidden_size,
            "weighting_hidden": args.weighting_hidden,
            "obs_dim": encoder.obs_dim,
            "num_actions": encoder.num_actions,
        },
        f"{args.out}/ensemble.pt",
    )
    with open(f"{args.out}/train_returns.txt", "w") as f:
        f.write("\n".join(str(r) for r in stats.returns))
    print(f"Trained adaptive ensemble on sub-policies {held_in} in {time.time() - t0:.1f}s")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
