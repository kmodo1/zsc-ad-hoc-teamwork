"""Recurrent actor-critic network shared by every trained coordination policy.

A GRU is used (rather than a plain MLP) so that a policy can maintain an
implicit notion of "which role am I playing this episode" -- this is what
lets independently-trained policies settle into different, internally
consistent conventions (e.g. always being the "onion runner"), which is
exactly the behavior the cross-play matrix in ``crossplay/`` is meant to
surface.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor_head = nn.Linear(hidden_size, num_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, rnn_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs: (B, obs_dim), rnn_state: (B, hidden_size).

        Returns (action_logits, value, new_rnn_state).
        """
        x = self.encoder(obs)
        new_rnn_state = self.gru(x, rnn_state)
        logits = self.actor_head(new_rnn_state)
        value = self.critic_head(new_rnn_state).squeeze(-1)
        return logits, value, new_rnn_state

    def act(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, log_prob, value, new_rnn_state)."""
        logits, value, new_rnn_state = self.forward(obs, rnn_state)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_rnn_state

    def action_probs(
        self, obs: torch.Tensor, rnn_state: torch.Tensor, available_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (probs over actions, new_rnn_state) without sampling."""
        logits, _value, new_rnn_state = self.forward(obs, rnn_state)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), -1e10)
        probs = torch.softmax(logits, dim=-1)
        return probs, new_rnn_state

    def evaluate_actions(
        self, obs: torch.Tensor, rnn_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO updates. Returns (log_prob, entropy, value, new_rnn_state)."""
        logits, value, new_rnn_state = self.forward(obs, rnn_state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, new_rnn_state
