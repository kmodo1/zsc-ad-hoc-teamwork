"""Maps a partner-trajectory embedding to a weight over each policy in the
ensemble (Section 4: "This representation will be passed to a weighting
network that outputs one weight for each policy in the ensemble").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightingNetwork(nn.Module):
    def __init__(self, embed_dim: int, num_policies: int, hidden_size: int = 32):
        super().__init__()
        self.num_policies = num_policies
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_policies),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns softmax weights, shape (B, num_policies), summing to 1."""
        logits = self.net(embedding)
        return torch.softmax(logits, dim=-1)
