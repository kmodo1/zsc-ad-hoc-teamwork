"""Static-ensemble baseline: combines the same frozen sub-policies as the
adaptive ensemble, but with fixed, uniform weights that never change based
on the partner. This isolates how much of the adaptive ensemble's benefit
(if any) actually comes from *adapting* its weights, as opposed to simply
averaging over a diverse population (Section 4: "compare the adaptive
ensemble against self-play, static ensembles, and policy-selection
baselines").
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic


class StaticEnsembleAgent(Agent):
    def __init__(
        self,
        sub_policies: List[ActorCritic],
        weights: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.sub_policies = sub_policies
        self.device = device
        self.weights = weights if weights is not None else np.full(len(sub_policies), 1.0 / len(sub_policies), dtype=np.float32)
        assert len(self.weights) == len(sub_policies)

    def reset(self) -> None:
        self._sub_rnn_states = [p.initial_state(1, self.device) for p in self.sub_policies]

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        combined = np.zeros(self.sub_policies[0].num_actions, dtype=np.float32)
        for k, policy in enumerate(self.sub_policies):
            probs, new_state = policy.action_probs(obs_t, self._sub_rnn_states[k])
            self._sub_rnn_states[k] = new_state
            combined += self.weights[k] * probs.squeeze(0).cpu().numpy()
        combined = combined / combined.sum()
        return combined

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        probs = self.action_probs(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
