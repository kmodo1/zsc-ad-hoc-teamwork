"""Presents a trained ZSC-Eval actor through the same inference interface as
:class:`~adhoc_teamwork.policies.networks.ActorCritic`, so ZSC-Eval's
populations can be used as ensemble sub-policies, partners and baselines
without the rest of this package knowing where they came from.

Requires the ``zsceval`` environment. Inputs are this package's flattened,
[0, 1]-scaled observations; they are restored to the raw ``(H, W, C)``,
255-scaled grid ZSC-Eval's actors were trained on.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ZSCEvalPolicy(nn.Module):
    def __init__(self, actor: nn.Module, obs_shape: Tuple[int, ...], num_actions: int, name: str = ""):
        super().__init__()
        self.actor = actor
        self.obs_shape = tuple(obs_shape)
        self.obs_dim = int(torch.tensor(self.obs_shape).prod().item())
        self.num_actions = int(num_actions)
        self.hidden_size = int(actor.hidden_size)
        self.name = name
        self.recurrent_n = int(getattr(actor, "_recurrent_N", 1))
        self.is_recurrent = bool(actor._use_recurrent_policy or actor._use_naive_recurrent_policy)

    def initial_state(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.recurrent_n, self.hidden_size, device=device)

    def action_probs(
        self, obs: torch.Tensor, rnn_state: torch.Tensor, available_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, obs_dim). Returns (probs (B, num_actions), new_rnn_state)."""
        raw_obs = obs.reshape(-1, *self.obs_shape) * 255.0
        features = self.actor.base(raw_obs)
        if self.is_recurrent:
            not_done = torch.ones(obs.shape[0], 1, device=obs.device)
            features, rnn_state = self.actor.rnn(features, rnn_state, not_done)
        if getattr(self.actor, "_layer_after_N", 0) > 0:
            features = self.actor.mlp_after(features)
        probs = self.actor.act.get_probs(features, available_actions)
        return probs, rnn_state


def load_zsceval_policy(pool_root: str, layout: str, rel_path: str, device: torch.device = torch.device("cpu"), name: str = "") -> ZSCEvalPolicy:
    """Load one actor checkpoint from a ZSC-Eval ``policy_pool`` directory.

    ``rel_path`` is relative to ``pool_root``. Whether it is the recurrent
    or feed-forward architecture is read off the checkpoint rather than
    assumed: in ZSC-Eval's pool, stage-1 populations are feed-forward while
    self-play and stage-2 agents are recurrent.
    """
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    path = os.path.join(pool_root, rel_path)
    # weights_only: checkpoints are plain tensors, and pool files may be downloaded.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    kind = "rnn" if any(k.startswith("rnn.") for k in state_dict) else "mlp"
    config_path = os.path.join(pool_root, layout, "policy_config", f"{kind}_policy_config.pkl")
    with open(config_path, "rb") as f:
        args, obs_space, _share_obs_space, act_space = pickle.load(f)
    actor = R_Actor(args, obs_space, act_space, device=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    return ZSCEvalPolicy(actor, obs_space.shape, act_space.n, name=name or rel_path).to(device)
