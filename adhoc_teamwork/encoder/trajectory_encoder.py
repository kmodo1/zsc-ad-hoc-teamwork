"""Partner trajectory encoder (Section 4 of the proposal): "the agent will
record the partner's trajectory, based on its observations and their
actions. A GRU-based encoder will convert this trajectory into a partner
representation."

Concretely, at each timestep the encoder consumes (the ego agent's own
observation at time t, the partner's action at time t-1) and updates a
recurrent embedding of "what kind of partner is this". Because it is a
GRU processed one step at a time, the representation is naturally
*online*: it keeps changing as more of the partner's behavior is observed
within an episode, which is what lets the downstream weighting network
(``ensemble/weighting_network.py``) re-weight the policy ensemble as play
continues (also required by the proposal: "these weights will change as
play with the partner continues over time").
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 32, embed_dim: int = 16, use_partner_actions: bool = True, use_memory: bool = True):
        """The two flags exist for ablations, testing whether a benefit really
        comes from observing the partner over time. Without either, the output
        can only depend on the current state: any gain left is state-based
        gating between sub-policies, not partner modeling.
        """
        super().__init__()
        self.use_partner_actions = use_partner_actions
        self.use_memory = use_memory
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # input = ego observation at time t ++ one-hot of the partner's
        # action at time t-1 (all-zero at the very first timestep, when no
        # partner action has been observed yet).
        self.input_proj = nn.Sequential(nn.Linear(obs_dim + num_actions, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.output_head = nn.Sequential(nn.Linear(hidden_size, embed_dim), nn.Tanh())

    def initial_state(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def initial_partner_action(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_actions, device=device)

    def step(
        self,
        ego_obs: torch.Tensor,
        prev_partner_action_onehot: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ego_obs: (B, obs_dim); prev_partner_action_onehot: (B, num_actions).

        Returns (embedding (B, embed_dim), new_rnn_state (B, hidden_size)).
        """
        if not self.use_partner_actions:
            prev_partner_action_onehot = torch.zeros_like(prev_partner_action_onehot)
        if not self.use_memory:
            rnn_state = torch.zeros_like(rnn_state)
        x = torch.cat([ego_obs, prev_partner_action_onehot], dim=-1)
        x = self.input_proj(x)
        new_rnn_state = self.gru(x, rnn_state)
        embedding = self.output_head(new_rnn_state)
        return embedding, new_rnn_state

    def action_to_onehot(self, action, device=None) -> torch.Tensor:
        """``action``: an int, or an array of shape (B,). Returns (B, num_actions)."""
        actions = torch.as_tensor(np.atleast_1d(action), dtype=torch.long, device=device)
        return F.one_hot(actions, num_classes=self.num_actions).float()
