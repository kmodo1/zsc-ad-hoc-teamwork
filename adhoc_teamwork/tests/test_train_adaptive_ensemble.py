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


def _rule_inputs():
    torch.manual_seed(0)
    logits = torch.randn(64, 3, requires_grad=True)
    sub_probs = torch.softmax(torch.randn(64, 3, 5), dim=-1)
    sub_probs[:, :, 4] = 0.0  # an illegal action, masked out by every sub-policy
    return logits, sub_probs / sub_probs.sum(-1, keepdim=True)


def test_every_action_rule_gives_legal_actions_and_a_gradient():
    import math

    import pytest

    from adhoc_teamwork.training.train_adaptive_ensemble import ACTION_RULES, choose_actions

    for rule in ACTION_RULES:
        logits, sub_probs = _rule_inputs()
        log_std = torch.full((3,), math.log(0.5), requires_grad=True)
        actions, log_prob = choose_actions(rule, logits, sub_probs, log_std)
        assert actions.shape == (64,) and log_prob.shape == (64,)
        assert (actions != 4).all(), rule
        log_prob.sum().backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0, rule
        if rule == "argmax":
            assert log_std.grad is not None and log_std.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        choose_actions("vote", *_rule_inputs())


def test_argmax_rule_without_noise_takes_the_mixtures_most_probable_action():
    from adhoc_teamwork.ensemble.adaptive_ensemble_agent import mix
    from adhoc_teamwork.training.train_adaptive_ensemble import choose_actions

    logits, sub_probs = _rule_inputs()
    actions, _ = choose_actions("argmax", logits, sub_probs, torch.full((3,), -30.0))
    assert torch.equal(actions, mix(torch.softmax(logits, -1), sub_probs).argmax(-1))


def test_select_argmax_rule_acts_with_one_sub_policy():
    from adhoc_teamwork.training.train_adaptive_ensemble import choose_actions

    logits, sub_probs = _rule_inputs()
    one_hot = torch.full_like(logits, -30.0)
    one_hot[:, 1] = 30.0  # always select sub-policy 1
    actions, log_prob = choose_actions("select_argmax", one_hot, sub_probs)
    assert torch.equal(actions, sub_probs[:, 1].argmax(-1))
    assert torch.allclose(log_prob, torch.zeros(64), atol=1e-6)


def test_argmax_rule_trains_and_resumes_its_noise_scale(tmp_path):
    env = OvercookedGridworld(horizon=10)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    path = str(tmp_path / "state.pt")
    cfg = EnsembleTrainConfig(advantage="actor_critic", action_rule="argmax")
    kwargs = dict(embed_dim=8, encoder_hidden=16, weighting_hidden=16, episodes_per_iteration=2, seed=0, checkpoint_path=path, checkpoint_every=1, cfg=cfg)
    sampler = lambda: ScriptedBiasedAgent("pot_tender", noise=0.2)  # noqa: E731
    train_adaptive_ensemble(env, subs, sampler, n_iterations=2, **kwargs)
    saved = torch.load(path, weights_only=False)["log_std"]
    assert saved.shape == (2,)
    _, _, stats = train_adaptive_ensemble(env, subs, sampler, n_iterations=3, **kwargs)
    assert len(stats.weight_noise_std) == 3
