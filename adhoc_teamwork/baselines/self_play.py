"""Self-play / single-policy baselines (Section 4: "Our baselines will be
algorithms such as: ... Multi-Agent Proximal Policy Optimization (Yu et al.,
2022), a general non-ZSC MARL algorithm that we can use as a baseline for
all performances").

There is no separate algorithm here: a self-play baseline *is* exactly one
member of the trained population (``population/policy_pool.py``), wrapped
for inference with :class:`NNAgent` and evaluated zero-shot against a
partner it never trained with. Our self-play PPO trainer already is a
(decentralized-critic) MAPPO-style algorithm, so the "self-play" and
"single-policy MAPPO" baselines from the proposal coincide in this
simplified codebase; a distinct centralized-critic MAPPO variant would be
a natural extension on the real Overcooked env (see ``envs/README`` /
``zsceval_adapter.py``) where observations and shared state actually
differ.
"""

from __future__ import annotations

import torch

from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.policies.nn_agent import NNAgent


def self_play_baseline_agent(policy: ActorCritic, device: torch.device = torch.device("cpu")) -> NNAgent:
    return NNAgent(policy, device=device)
