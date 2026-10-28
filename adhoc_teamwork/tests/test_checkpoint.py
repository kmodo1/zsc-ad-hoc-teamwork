"""Regression test for ensemble/checkpoint.py: the weighting network must
be reconstructed with the number of sub-policies it was actually *trained*
with (``len(held_in)``), which can be a strict subset of the full
population pool -- not whatever size the caller happens to pass in.
"""

import os

from adhoc_teamwork.ensemble.checkpoint import load_ensemble_checkpoint
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.training.train_adaptive_ensemble import EnsembleTrainConfig, train_adaptive_ensemble
from adhoc_teamwork.policies.networks import ActorCritic
import torch


def test_load_checkpoint_with_held_in_subset_of_population(tmp_path):
    env = OvercookedGridworld(horizon=10)
    full_population = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(3)]
    held_in = full_population[:2]  # only 2 of the 3 population policies used as sub-policies

    encoder, weighting_net, _ = train_adaptive_ensemble(
        env,
        held_in,
        partner_sampler=lambda: ScriptedBiasedAgent("onion_runner", noise=0.1),
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=1,
        episodes_per_iteration=1,
        cfg=EnsembleTrainConfig(),
        seed=0,
    )
    ckpt_path = os.path.join(tmp_path, "ensemble.pt")
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "weighting_state_dict": weighting_net.state_dict(),
            "held_in": ["policy_0", "policy_1"],
            "embed_dim": 8,
            "encoder_hidden": 16,
            "weighting_hidden": 16,
            "obs_dim": encoder.obs_dim,
            "num_actions": encoder.num_actions,
        },
        ckpt_path,
    )

    loaded_encoder, loaded_weighting_net, held_in_names = load_ensemble_checkpoint(str(ckpt_path))
    assert held_in_names == ["policy_0", "policy_1"]
    assert loaded_weighting_net.num_policies == 2
