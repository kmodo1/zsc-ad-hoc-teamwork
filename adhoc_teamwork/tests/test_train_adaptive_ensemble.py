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


def test_actor_critic_training_changes_encoder_params():
    env = OvercookedGridworld(horizon=15)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    encoder_init, _, _ = train_adaptive_ensemble(env, subs, lambda: ScriptedBiasedAgent("pot_tender", noise=0.2), embed_dim=8, encoder_hidden=16, weighting_hidden=16, n_iterations=0, seed=0)
    encoder, _, stats = train_adaptive_ensemble(
        env,
        subs,
        lambda: ScriptedBiasedAgent("pot_tender", noise=0.2),
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=2,
        episodes_per_iteration=2,
        cfg=EnsembleTrainConfig(advantage="actor_critic"),
        seed=0,
    )
    assert len(stats.returns) == 2
    assert any(not torch.allclose(a, b) for a, b in zip(encoder_init.parameters(), encoder.parameters()))


def test_training_resumes_from_checkpoint(tmp_path):
    env = OvercookedGridworld(horizon=10)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    path = str(tmp_path / "state.pt")
    kwargs = dict(embed_dim=8, encoder_hidden=16, weighting_hidden=16, episodes_per_iteration=2, seed=0, checkpoint_path=path, checkpoint_every=1)
    sampler = lambda: ScriptedBiasedAgent("pot_tender", noise=0.2)  # noqa: E731
    train_adaptive_ensemble(env, subs, sampler, n_iterations=2, **kwargs)
    assert torch.load(path, weights_only=False)["iteration"] == 2
    _, _, stats = train_adaptive_ensemble(env, subs, sampler, n_iterations=3, **kwargs)
    assert len(stats.returns) == 3, "should continue from iteration 2, keeping the earlier history"
    assert torch.load(path, weights_only=False)["iteration"] == 3


def test_partner_assignment_covers_every_env_and_balances_positions():
    import numpy as np

    from adhoc_teamwork.training.train_adaptive_ensemble import assign_partners

    a, b = ScriptedBiasedAgent("pot_tender"), ScriptedBiasedAgent("onion_runner")
    groups, ego_positions = assign_partners([a, b, a, b], 8)  # duplicates merge
    assert len(groups) == 2
    assert sorted(np.concatenate([idx for _, idx in groups]).tolist()) == list(range(8))
    for _, idx in groups:
        assert ego_positions[idx].sum() == len(idx) // 2


def test_gae_matches_hand_computation():
    import numpy as np

    from adhoc_teamwork.training.train_adaptive_ensemble import gae

    rewards = np.array([[0.0], [1.0]], dtype=np.float32)
    values = np.array([[0.5], [0.25]], dtype=np.float32)
    advantages, returns = gae(rewards, values, gamma=0.5, lam=0.5)
    # t=1: delta = 1 - 0.25 = 0.75. t=0: delta = 0 + 0.5*0.25 - 0.5 = -0.375; adv = -0.375 + 0.25*0.75
    assert np.allclose(advantages[:, 0], [-0.1875, 0.75])
    assert np.allclose(returns, advantages + values)
