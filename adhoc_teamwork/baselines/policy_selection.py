"""Policy-selection baseline: hard (argmax) analogue of the adaptive
ensemble, in the spirit of Barrett et al. (2017)'s "select an optimal
action from a set of pre-trained policies" and CBPR (Wang et al., 2024a)'s
"adaptively selects one [policy] based on the current partner". Instead of
continuously *blending* every sub-policy's action distribution (as
``AdaptiveEnsembleAgent`` does), this baseline commits each step to a
single sub-policy -- whichever one the (same, already-trained) partner
encoder + weighting network currently scores highest -- and acts only
according to that policy.

Reusing the trained encoder/weighting network (rather than training a
wholly separate selection mechanism) keeps the comparison to the adaptive
ensemble controlled: the only thing that differs is soft mixing vs. hard
selection of the same learned partner-compatibility signal.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder
from adhoc_teamwork.ensemble.weighting_network import WeightingNetwork
from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic


class PolicySelectionAgent(Agent):
    def __init__(
        self,
        sub_policies: List[ActorCritic],
        encoder: TrajectoryEncoder,
        weighting_net: WeightingNetwork,
        device: torch.device = torch.device("cpu"),
    ):
        self.sub_policies = sub_policies
        self.encoder = encoder
        self.weighting_net = weighting_net
        self.device = device
        self.last_selected = None
        self.last_weights = None

    def reset(self) -> None:
        self._sub_rnn_states = [p.initial_state(1, self.device) for p in self.sub_policies]
        self._encoder_state = self.encoder.initial_state(1, self.device)
        self._prev_partner_onehot = self.encoder.initial_partner_action(1, self.device)
        self.last_selected = None
        self.last_weights = None

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        embedding, self._encoder_state = self.encoder.step(obs_t, self._prev_partner_onehot, self._encoder_state)
        weights = self.weighting_net(embedding).squeeze(0).cpu().numpy()
        self.last_weights = weights
        selected = int(np.argmax(weights))
        self.last_selected = selected

        # Every sub-policy's hidden state still has to advance every step
        # (even if not currently selected) so that whichever one gets
        # selected later has a consistent internal state.
        probs_out = None
        for k, policy in enumerate(self.sub_policies):
            probs, new_state = policy.action_probs(obs_t, self._sub_rnn_states[k])
            self._sub_rnn_states[k] = new_state
            if k == selected:
                probs_out = probs.squeeze(0).cpu().numpy()
        return probs_out

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        probs = self.action_probs(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def observe_partner_action(self, partner_action: int) -> None:
        self._prev_partner_onehot = self.encoder.action_to_onehot(partner_action, device=self.device)
