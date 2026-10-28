"""Save/load helpers for a trained (encoder, weighting_net) pair."""

from __future__ import annotations

import torch

from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder
from adhoc_teamwork.ensemble.weighting_network import WeightingNetwork


def load_ensemble_checkpoint(path: str, num_policies: int = None):
    """``num_policies`` is accepted for backwards-compatible call sites but
    ignored: the weighting network's output size must match the number of
    sub-policies it was *trained* with (``len(held_in)``), which is saved
    in the checkpoint itself and may be smaller than the full population.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = TrajectoryEncoder(ckpt["obs_dim"], ckpt["num_actions"], hidden_size=ckpt["encoder_hidden"], embed_dim=ckpt["embed_dim"])
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    weighting_net = WeightingNetwork(ckpt["embed_dim"], len(ckpt["held_in"]), hidden_size=ckpt.get("weighting_hidden", 32))
    weighting_net.load_state_dict(ckpt["weighting_state_dict"])
    return encoder, weighting_net, ckpt["held_in"]
