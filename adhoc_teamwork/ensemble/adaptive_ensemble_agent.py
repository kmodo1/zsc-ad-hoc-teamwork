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

from typing import List

import numpy as np
import torch

from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder
from adhoc_teamwork.ensemble.weighting_network import WeightingNetwork
from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic


class AdaptiveEnsembleAgent(Agent):
    def __init__(
        self,
        sub_policies: List[ActorCritic],
        encoder: TrajectoryEncoder,
        weighting_net: WeightingNetwork,
        device: torch.device = torch.device("cpu"),
        deterministic_weights: bool = True,
    ):
        assert len(sub_policies) == weighting_net.num_policies
        self.sub_policies = sub_policies
        self.encoder = encoder
        self.weighting_net = weighting_net
        self.device = device
        self.deterministic_weights = deterministic_weights
        self.last_weights = None

    def reset(self) -> None:
        self._sub_rnn_states = [p.initial_state(1, self.device) for p in self.sub_policies]
        self._encoder_state = self.encoder.initial_state(1, self.device)
        self._prev_partner_onehot = self.encoder.initial_partner_action(1, self.device)
        self.last_weights = None

    @torch.no_grad()
    def _weights_and_sub_probs(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        embedding, new_encoder_state = self.encoder.step(obs_t, self._prev_partner_onehot, self._encoder_state)
        self._encoder_state = new_encoder_state
        weights = self.weighting_net(embedding).squeeze(0)  # (num_policies,)

        sub_probs = []
        for k, policy in enumerate(self.sub_policies):
            probs, new_state = policy.action_probs(obs_t, self._sub_rnn_states[k])
            self._sub_rnn_states[k] = new_state
            sub_probs.append(probs.squeeze(0))
        return weights, sub_probs

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        weights, sub_probs = self._weights_and_sub_probs(obs)
        self.last_weights = weights.cpu().numpy()
        combined = sum(w * p for w, p in zip(weights, sub_probs))
        combined = combined / combined.sum()
        if deterministic:
            return int(torch.argmax(combined).item())
        return int(torch.multinomial(combined, 1).item())

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        weights, sub_probs = self._weights_and_sub_probs(obs)
        self.last_weights = weights.cpu().numpy()
        combined = sum(w * p for w, p in zip(weights, sub_probs))
        combined = combined / combined.sum()
        return combined.cpu().numpy()

    def observe_partner_action(self, partner_action: int) -> None:
        self._prev_partner_onehot = self.encoder.action_to_onehot(partner_action, device=self.device)
