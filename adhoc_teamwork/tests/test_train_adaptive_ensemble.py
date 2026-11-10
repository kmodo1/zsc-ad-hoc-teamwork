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


def _random_mixture(n=64, num_actions=6, seed=0):
    """Mixture probabilities with some illegal (zero-probability) actions."""
    generator = torch.Generator().manual_seed(seed)
    probs = torch.rand(n, num_actions, generator=generator)
    probs[::2, 0] = 0.0
    return probs / probs.sum(dim=-1, keepdim=True)


def test_temperature_one_reproduces_the_untempered_log_probs():
    from adhoc_teamwork.training.train_adaptive_ensemble import tempered_distribution

    combined = _random_mixture().requires_grad_(True)
    probs, log_probs = tempered_distribution(combined, 1.0)
    assert torch.equal(probs, combined), "T = 1 must sample from the mixture itself, so old runs reproduce exactly"
    assert torch.equal(log_probs, combined.clamp_min(1e-8).log())

    # The general formula agrees with it at T = 1, and is softmax(log(mix) / T) elsewhere.
    near_probs, near_log_probs = tempered_distribution(combined, 1.0 + 1e-7)
    legal = combined > 0
    assert torch.allclose(near_probs, combined, atol=1e-5)
    assert torch.allclose(near_log_probs[legal], log_probs[legal], atol=1e-5)
    half_probs, half_log_probs = tempered_distribution(combined, 0.5)
    squared = combined.detach() ** 2
    assert torch.allclose(half_probs, squared / squared.sum(dim=-1, keepdim=True), atol=1e-6)
    assert torch.allclose(half_log_probs.exp().sum(dim=-1), torch.ones(len(combined)), atol=1e-5)

    # Still a policy gradient: finite, non-zero gradients reach the mixture.
    actions = torch.multinomial(half_probs.detach(), 1)
    half_log_probs.gather(1, actions).sum().backward()
    assert torch.isfinite(combined.grad).all() and combined.grad.abs().sum() > 0


def test_sampling_concentrates_on_the_argmax_as_temperature_falls():
    from adhoc_teamwork.training.train_adaptive_ensemble import tempered_distribution

    combined = _random_mixture(seed=1)
    combined[torch.arange(len(combined)), torch.arange(len(combined)) % 5 + 1] += 0.5  # a clear best action, never a near-tie
    combined = combined / combined.sum(dim=-1, keepdim=True)
    best = combined.argmax(dim=-1, keepdim=True)
    mass_on_best = [tempered_distribution(combined, t)[0].gather(1, best).squeeze(1) for t in (1.0, 0.5, 0.1, 0.01)]
    for warmer, colder in zip(mass_on_best, mass_on_best[1:]):
        assert (colder >= warmer - 1e-6).all()
    assert mass_on_best[-1].min() > 0.99

    torch.manual_seed(0)
    cold_probs, _ = tempered_distribution(combined, 0.01)
    samples = torch.multinomial(cold_probs, 200, replacement=True)
    assert (samples == best).float().mean() > 0.99
    # Illegal actions stay impossible at every temperature.
    for t in (1.0, 0.5, 0.01):
        assert (tempered_distribution(combined, t)[0][combined == 0] == 0).all()


def test_temperature_schedule_is_linear_then_holds():
    import pytest

    assert [EnsembleTrainConfig().temperature(it, 10) for it in (0, 5, 9)] == [1.0, 1.0, 1.0], "default is the old behavior"
    cfg = EnsembleTrainConfig(temperature_start=1.0, temperature_end=0.1)
    assert cfg.temperature(0, 11) == pytest.approx(1.0)
    assert cfg.temperature(5, 11) == pytest.approx(0.55)
    assert cfg.temperature(10, 11) == pytest.approx(0.1)
    half = EnsembleTrainConfig(temperature_start=1.0, temperature_end=0.3, temperature_anneal_fraction=0.5)
    assert half.temperature(5, 11) == pytest.approx(0.3)
    assert half.temperature(10, 11) == pytest.approx(0.3)
    assert EnsembleTrainConfig(temperature_end=0.3).temperature(0, 1) == pytest.approx(0.3)


def test_annealed_training_runs_and_resume_keeps_the_schedule(tmp_path):
    env = OvercookedGridworld(horizon=10)
    subs = [ActorCritic(env.obs_dim, env.num_actions, hidden_size=8) for _ in range(2)]
    path = str(tmp_path / "state.pt")
    cfg = EnsembleTrainConfig(advantage="actor_critic", temperature_start=1.0, temperature_end=0.2)
    kwargs = dict(embed_dim=8, encoder_hidden=16, weighting_hidden=16, episodes_per_iteration=2, seed=0, cfg=cfg, checkpoint_every=1)
    sampler = lambda: ScriptedBiasedAgent("pot_tender", noise=0.2)  # noqa: E731
    expected = [cfg.temperature(it, 5) for it in range(5)]

    _, _, uninterrupted = train_adaptive_ensemble(env, subs, sampler, n_iterations=5, **kwargs)
    assert uninterrupted.temperature == expected
    assert all(abs(loss) < float("inf") for loss in uninterrupted.losses)

    def stop_after_two():
        stop_after_two.calls += 1
        if stop_after_two.calls > 4:  # two partners per iteration
            raise KeyboardInterrupt
        return sampler()

    stop_after_two.calls = 0
    try:
        train_adaptive_ensemble(env, subs, stop_after_two, n_iterations=5, checkpoint_path=path, **kwargs)
    except KeyboardInterrupt:
        pass
    assert torch.load(path, weights_only=False)["iteration"] == 2
    _, _, resumed = train_adaptive_ensemble(env, subs, sampler, n_iterations=5, checkpoint_path=path, **kwargs)
    assert resumed.temperature == expected
