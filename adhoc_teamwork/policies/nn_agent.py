"""Wraps a trained policy network for inference, implementing the common
:class:`Agent` interface. Works with anything exposing ``initial_state`` and
``action_probs``: our :class:`ActorCritic` or a ZSC-Eval actor via
:class:`~adhoc_teamwork.policies.zsceval_policy.ZSCEvalPolicy`.
"""

from __future__ import annotations

import numpy as np
import torch

from adhoc_teamwork.policies.agent_api import TorchPolicyAgent


class NNAgent(TorchPolicyAgent):
    def __init__(self, policy, device: torch.device = torch.device("cpu"), argmax_actions: bool = False):
        super().__init__(device, argmax_actions)
        self.policy = policy
        self.rnn_state = None

    def reset(self, batch_size: int = 1) -> None:
        self.rnn_state = self.policy.initial_state(batch_size, self.device)

    def _forward_probs(self, obs_t, available_t):
        probs, self.rnn_state = self.policy.action_probs(obs_t, self.rnn_state, available_t)
        return probs

    def act_with_probs(self, obs: np.ndarray, deterministic: bool = False):
        probs = self.action_probs(obs)
        action = int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs / probs.sum()))
        return action, probs
