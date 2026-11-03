"""The adaptive ensemble agent: combines a frozen population of
coordination policies into a single agent whose mixture weights are
continuously re-computed from a trajectory-conditioned embedding of the
current partner (Section 4: "The final distribution will be the result of
a weighted combination of the sub-policies' outputs ... These weights will
change as play with the partner continues over time").

This class is inference-only (no gradients) -- see
``training/train_adaptive_ensemble.py`` for how the encoder and weighting
network are actually trained.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder
from adhoc_teamwork.ensemble.weighting_network import WeightingNetwork
from adhoc_teamwork.policies.agent_api import TorchPolicyAgent


def sub_policy_probs(sub_policies, rnn_states: List[torch.Tensor], obs_t: torch.Tensor, available_t: Optional[torch.Tensor]) -> torch.Tensor:
    """Every sub-policy's action distribution for this step, shape
    (B, K, num_actions). Advances ``rnn_states`` in place; every sub-policy
    must advance every step so that its internal state stays consistent
    whether or not it is currently influencing the action.
    """
    probs = []
    for k, policy in enumerate(sub_policies):
        p, rnn_states[k] = policy.action_probs(obs_t, rnn_states[k], available_t)
        probs.append(p)
    return torch.stack(probs, dim=1)


def mix(weights: torch.Tensor, sub_probs: torch.Tensor) -> torch.Tensor:
    """weights (B, K), sub_probs (B, K, A) -> mixture distribution (B, A)."""
    combined = (weights.unsqueeze(-1) * sub_probs).sum(dim=1)
    return combined / combined.sum(dim=-1, keepdim=True)


class TrajectoryConditionedAgent(TorchPolicyAgent):
    """Shared machinery for agents that run the partner encoder + weighting
    network online: the adaptive ensemble and the policy-selection baseline.
    """

    def __init__(self, sub_policies, encoder: TrajectoryEncoder, weighting_net: WeightingNetwork, device: torch.device = torch.device("cpu"), argmax_actions: bool = False):
        super().__init__(device, argmax_actions)
        assert len(sub_policies) == weighting_net.num_policies
        self.sub_policies = sub_policies
        self.encoder = encoder
        self.weighting_net = weighting_net
        self.last_weights = None

    def reset(self, batch_size: int = 1) -> None:
        self._sub_rnn_states = [p.initial_state(batch_size, self.device) for p in self.sub_policies]
        self._encoder_state = self.encoder.initial_state(batch_size, self.device)
        self._prev_partner_onehot = self.encoder.initial_partner_action(batch_size, self.device)
        self.last_weights = None

    def _weights_and_sub_probs(self, obs_t, available_t):
        embedding, self._encoder_state = self.encoder.step(obs_t, self._prev_partner_onehot, self._encoder_state)
        weights = self.weighting_net(embedding)
        sub_probs = sub_policy_probs(self.sub_policies, self._sub_rnn_states, obs_t, available_t)
        # (K,) for a single episode, (B, K) for a batch.
        self.last_weights = weights.squeeze(0).cpu().numpy() if weights.shape[0] == 1 else weights.cpu().numpy()
        return weights, sub_probs

    def observe_partner_action(self, partner_action) -> None:
        self._prev_partner_onehot = self.encoder.action_to_onehot(partner_action, device=self.device)


class AdaptiveEnsembleAgent(TrajectoryConditionedAgent):
    def _forward_probs(self, obs_t, available_t):
        weights, sub_probs = self._weights_and_sub_probs(obs_t, available_t)
        return mix(weights, sub_probs)
