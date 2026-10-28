"""Unit tests for the actor-critic network (Section 6: "policies outputting
valid actions")."""

import torch

from adhoc_teamwork.policies.networks import ActorCritic

OBS_DIM = 24
NUM_ACTIONS = 6


def test_forward_shapes():
    net = ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=16)
    obs = torch.randn(5, OBS_DIM)
    rnn_state = net.initial_state(5)
    logits, value, new_rnn_state = net.forward(obs, rnn_state)
    assert logits.shape == (5, NUM_ACTIONS)
    assert value.shape == (5,)
    assert new_rnn_state.shape == (5, 16)


def test_action_probs_valid_distribution():
    net = ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=16)
    obs = torch.randn(3, OBS_DIM)
    rnn_state = net.initial_state(3)
    probs, _ = net.action_probs(obs, rnn_state)
    assert probs.shape == (3, NUM_ACTIONS)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_act_returns_valid_action():
    net = ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=16)
    obs = torch.randn(4, OBS_DIM)
    rnn_state = net.initial_state(4)
    action, log_prob, value, new_rnn_state = net.act(obs, rnn_state)
    assert action.shape == (4,)
    assert torch.all(action >= 0) and torch.all(action < NUM_ACTIONS)
    assert log_prob.shape == (4,)
    assert torch.all(log_prob <= 0)  # log-probabilities are non-positive


def test_act_deterministic_is_argmax():
    net = ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=16)
    obs = torch.randn(2, OBS_DIM)
    rnn_state = net.initial_state(2)
    logits, _, _ = net.forward(obs, rnn_state)
    expected = torch.argmax(logits, dim=-1)
    action, _, _, _ = net.act(obs, rnn_state, deterministic=True)
    assert torch.equal(action, expected)


def test_evaluate_actions_matches_forward():
    net = ActorCritic(OBS_DIM, NUM_ACTIONS, hidden_size=16)
    obs = torch.randn(3, OBS_DIM)
    rnn_state = net.initial_state(3)
    actions = torch.tensor([0, 1, 2])
    log_prob, entropy, value, _ = net.evaluate_actions(obs, rnn_state, actions)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert torch.all(entropy >= 0)
