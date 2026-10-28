"""Wraps a trained :class:`ActorCritic` network for inference, implementing
the common :class:`Agent` interface.
"""

from __future__ import annotations

import numpy as np
import torch

from adhoc_teamwork.policies.agent_api import Agent
from adhoc_teamwork.policies.networks import ActorCritic


class NNAgent(Agent):
    def __init__(self, policy: ActorCritic, device: torch.device = torch.device("cpu")):
        self.policy = policy
        self.device = device
        self.rnn_state = None

    def reset(self) -> None:
        self.rnn_state = self.policy.initial_state(1, self.device)

    def _probs_and_advance(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, new_rnn_state = self.policy.action_probs(obs_t, self.rnn_state)
        self.rnn_state = new_rnn_state
        return probs.squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        probs = self._probs_and_advance(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        # NOTE: this advances the recurrent state (as does `act`). Callers
        # that need both the probs *and* a sampled action for the same
        # timestep should use `act_with_probs` instead of calling both.
        return self._probs_and_advance(obs)

    def act_with_probs(self, obs: np.ndarray, deterministic: bool = False):
        probs = self._probs_and_advance(obs)
        action = int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))
        return action, probs
