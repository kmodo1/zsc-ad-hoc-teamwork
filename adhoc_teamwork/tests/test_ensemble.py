"""Unit tests for the weighting network, static ensemble, policy-selection
baseline, and adaptive ensemble agent."""

import numpy as np
import torch

from adhoc_teamwork.baselines.policy_selection import PolicySelectionAgent
from adhoc_teamwork.baselines.static_ensemble import StaticEnsembleAgent
from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder
from adhoc_teamwork.ensemble.adaptive_ensemble_agent import AdaptiveEnsembleAgent
from adhoc_teamwork.ensemble.weighting_network import WeightingNetwork
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.networks import ActorCritic

OBS_DIM = 24
NUM_ACTIONS = 6
K = 3


def test_weighting_network_outputs_valid_distribution():
    net = WeightingNetwork(embed_dim=8, num_policies=K)
    embedding = torch.randn(5, 8)
    weights = net(embedding)
    assert weights.shape == (5, K)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)


def _make_subpolicies(n=K):
    return [ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=8) for _ in range(n)]


def test_static_ensemble_action_probs_valid():
    agent = StaticEnsembleAgent(_make_subpolicies())
    agent.reset()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    probs = agent.action_probs(obs)
    assert probs.shape == (NUM_ACTIONS,)
    assert np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_static_ensemble_uses_uniform_weights_by_default():
    agent = StaticEnsembleAgent(_make_subpolicies())
    assert np.allclose(agent.weights, 1.0 / K)


def test_adaptive_ensemble_action_probs_valid_and_weights_sum_to_one():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    agent = AdaptiveEnsembleAgent(subs, encoder, wnet)
    agent.reset()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    probs = agent.action_probs(obs)
    assert probs.shape == (NUM_ACTIONS,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert agent.last_weights is not None
    assert np.isclose(agent.last_weights.sum(), 1.0, atol=1e-5)


def test_adaptive_ensemble_weights_change_with_partner_observations():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    agent = AdaptiveEnsembleAgent(subs, encoder, wnet)
    agent.reset()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    agent.act(obs)
    w0 = agent.last_weights.copy()
    for a in range(NUM_ACTIONS):
        agent.observe_partner_action(a)
        agent.act(obs)
    w1 = agent.last_weights.copy()
    assert not np.allclose(w0, w1)


def test_policy_selection_selects_exactly_one_policy():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    agent = PolicySelectionAgent(subs, encoder, wnet)
    agent.reset()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    probs = agent.action_probs(obs)
    assert probs.shape == (NUM_ACTIONS,)
    assert agent.last_selected in range(K)


def test_agents_run_full_episode_without_error():
    env = OvercookedGridworld(horizon=15)
    from adhoc_teamwork.evaluation.rollout import run_episode
    from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent

    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    for AgentCls, extra in [
        (StaticEnsembleAgent, {}),
        (AdaptiveEnsembleAgent, {"encoder": encoder, "weighting_net": wnet}),
        (PolicySelectionAgent, {"encoder": encoder, "weighting_net": wnet}),
    ]:
        agent = AgentCls(subs, **extra) if extra else AgentCls(subs)
        traj = run_episode(env, [agent, ScriptedBiasedAgent("onion_runner", noise=0.1)], seed=0)
        assert traj.length == 15


def test_batched_and_single_episode_inference_agree():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    make_agents = {
        "adaptive": lambda: AdaptiveEnsembleAgent(subs, encoder, wnet),
        "selection": lambda: PolicySelectionAgent(subs, encoder, wnet),
        "static": lambda: StaticEnsembleAgent(subs),
    }
    obs_seq = np.random.randn(5, 3, OBS_DIM).astype(np.float32)  # (T, B, obs_dim)
    actions = np.random.randint(0, NUM_ACTIONS, size=(5, 3))
    for name, make in make_agents.items():
        batched = make()
        batched.reset(batch_size=3)
        singles = [make() for _ in range(3)]
        for a in singles:
            a.reset()
        for t in range(5):
            batch_probs = batched.probs_batch(obs_seq[t]).numpy()
            batched.observe_partner_action(actions[t])
            for b, a in enumerate(singles):
                assert np.allclose(batch_probs[b], a.action_probs(obs_seq[t, b]), atol=1e-5), f"{name} batch element {b} at t={t}"
                a.observe_partner_action(int(actions[t, b]))


def test_argmax_actions_takes_most_probable_action():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    probe = AdaptiveEnsembleAgent(subs, encoder, wnet)
    probe.reset()
    expected = int(np.argmax(probe.action_probs(obs)))
    agent = AdaptiveEnsembleAgent(subs, encoder, wnet, argmax_actions=True)
    agent.reset()
    assert agent.act(obs) == expected


def test_policy_selection_uses_only_the_selected_sub_policy():
    subs = _make_subpolicies()
    encoder = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=16, embed_dim=8)
    wnet = WeightingNetwork(embed_dim=8, num_policies=K)
    agent = PolicySelectionAgent(subs, encoder, wnet)
    agent.reset()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    probs = agent.action_probs(obs)
    selected = int(agent.last_selected[0])
    assert selected == int(np.argmax(agent.last_weights))
    sub_probs, _ = subs[selected].action_probs(torch.as_tensor(obs)[None], subs[selected].initial_state(1))
    assert np.allclose(probs, sub_probs[0].detach().numpy(), atol=1e-6)
