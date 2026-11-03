"""Static-ensemble baseline: combines the same frozen sub-policies as the
adaptive ensemble, but with fixed, uniform weights that never change based
on the partner. This isolates how much of the adaptive ensemble's benefit
(if any) actually comes from *adapting* its weights, as opposed to simply
averaging over a diverse population (Section 4: "compare the adaptive
ensemble against self-play, static ensembles, and policy-selection
baselines").
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from adhoc_teamwork.ensemble.adaptive_ensemble_agent import mix, sub_policy_probs
from adhoc_teamwork.policies.agent_api import TorchPolicyAgent


class StaticEnsembleAgent(TorchPolicyAgent):
    def __init__(self, sub_policies, weights: Optional[np.ndarray] = None, device: torch.device = torch.device("cpu"), argmax_actions: bool = False):
        super().__init__(device, argmax_actions)
        self.sub_policies = sub_policies
        self.weights = weights if weights is not None else np.full(len(sub_policies), 1.0 / len(sub_policies), dtype=np.float32)
        assert len(self.weights) == len(sub_policies)
        self._weights_t = torch.as_tensor(self.weights, dtype=torch.float32, device=device).unsqueeze(0)

    def reset(self, batch_size: int = 1) -> None:
        self._sub_rnn_states = [p.initial_state(batch_size, self.device) for p in self.sub_policies]

    def _forward_probs(self, obs_t, available_t):
        sub_probs = sub_policy_probs(self.sub_policies, self._sub_rnn_states, obs_t, available_t)
        return mix(self._weights_t.expand(obs_t.shape[0], -1), sub_probs)
