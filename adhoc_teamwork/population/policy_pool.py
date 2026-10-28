"""A pool of independently self-play-trained coordination policies.

Population diversity here comes from training several self-play PPO agents
with different random seeds (see ``training/ppo_trainer.py``): because the
two agent roles in the gridworld are symmetric, independently-trained pairs
settle on different, often-incompatible conventions about who fetches
onions and who tends the pot -- this *is* the ZSC problem (Carroll et al.,
2019; Hu et al., 2020), and it is what makes the resulting population a
meaningful testbed for the adaptive ensemble.

This keeps the same role as ``zsceval``'s ``PolicyPool`` (register/save/load
a named set of policies) but is deliberately minimal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import torch

from adhoc_teamwork.envs.base import TwoAgentEnv
from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.training.ppo_trainer import PPOConfig, train_self_play


@dataclass
class PolicyPool:
    policies: Dict[str, ActorCritic]

    def __init__(self):
        self.policies = {}

    def add(self, name: str, policy: ActorCritic):
        self.policies[name] = policy

    def names(self) -> List[str]:
        return list(self.policies.keys())

    def __getitem__(self, name: str) -> ActorCritic:
        return self.policies[name]

    def __len__(self):
        return len(self.policies)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, policy in self.policies.items():
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "obs_dim": policy.obs_dim,
                    "num_actions": policy.num_actions,
                    "hidden_size": policy.hidden_size,
                },
                os.path.join(path, f"{name}.pt"),
            )

    @classmethod
    def load(cls, path: str) -> "PolicyPool":
        pool = cls()
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".pt"):
                continue
            ckpt = torch.load(os.path.join(path, fname), map_location="cpu", weights_only=False)
            policy = ActorCritic(ckpt["obs_dim"], ckpt["num_actions"], ckpt["hidden_size"])
            policy.load_state_dict(ckpt["state_dict"])
            pool.add(fname[: -len(".pt")], policy)
        return pool


def train_population(
    env: TwoAgentEnv,
    n_policies: int = 4,
    n_iterations: int = 150,
    episodes_per_iteration: int = 6,
    cfg: PPOConfig = None,
    base_seed: int = 0,
    verbose: bool = False,
) -> PolicyPool:
    """Train ``n_policies`` independent self-play PPO policies and return
    them as a :class:`PolicyPool`. This is the "train several diverse
    coordination policies" step of Section 3 of the proposal.
    """
    cfg = cfg or PPOConfig()
    pool = PolicyPool()
    for k in range(n_policies):
        seed = base_seed + k
        policy = train_self_play(
            env,
            n_iterations=n_iterations,
            episodes_per_iteration=episodes_per_iteration,
            cfg=cfg,
            device=torch.device("cpu"),
            seed=seed,
            verbose=verbose,
        )
        pool.add(f"policy_{k}", policy)
    return pool
