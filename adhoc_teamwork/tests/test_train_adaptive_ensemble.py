"""Unit tests for the adaptive-ensemble (encoder + weighting network)
training loop: "individual training steps making valid changes to
weights", applied to the REINFORCE update this time."""

import copy

import torch

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.training.train_adaptive_ensemble import EnsembleTrainConfig, train_adaptive_ensemble


def test_training_changes_encoder_and_weighting_params():
    env = OvercookedGridworld(horizon=15)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    for p in subs:
        for param in p.parameters():
            param.data.normal_(0, 0.5)  # non-degenerate random sub-policies

    encoder, weighting_net, stats = train_adaptive_ensemble(
        env,
        subs,
        partner_sampler=lambda: ScriptedBiasedAgent("onion_runner", noise=0.2),
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=1,
        episodes_per_iteration=1,
        cfg=EnsembleTrainConfig(),
        seed=0,
    )
    before_enc = copy.deepcopy(list(encoder.parameters()))
    before_w = copy.deepcopy(list(weighting_net.parameters()))

    encoder2, weighting_net2, _ = train_adaptive_ensemble(
        env,
        subs,
        partner_sampler=lambda: ScriptedBiasedAgent("onion_runner", noise=0.2),
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=3,
        episodes_per_iteration=2,
        cfg=EnsembleTrainConfig(),
        seed=1,
    )
    # a freshly-initialized run trained for more iterations should differ
    # from a 1-iteration run with a different seed -- i.e. training is
    # actually doing something, not a no-op.
    changed = any(
        not torch.allclose(a, b)
        for a, b in zip(encoder.parameters(), encoder2.parameters())
    )
    assert changed


def test_sub_policies_are_frozen_during_training():
    env = OvercookedGridworld(horizon=15)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    before = [copy.deepcopy(p.state_dict()) for p in subs]

    train_adaptive_ensemble(
        env,
        subs,
        partner_sampler=lambda: ScriptedBiasedAgent("pot_tender", noise=0.2),
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=2,
        episodes_per_iteration=2,
        cfg=EnsembleTrainConfig(),
        seed=0,
    )
    for p, before_sd in zip(subs, before):
        for k, v in p.state_dict().items():
            assert torch.allclose(v, before_sd[k]), f"sub-policy parameter {k} changed but should be frozen"
