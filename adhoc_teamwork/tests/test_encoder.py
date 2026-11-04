"""Unit tests for the partner trajectory encoder (Section 6: "trajectory
encoders producing valid embeddings")."""

import numpy as np
import torch

from adhoc_teamwork.encoder.trajectory_encoder import TrajectoryEncoder

OBS_DIM = 24
NUM_ACTIONS = 6
EMBED_DIM = 16


def test_encoder_step_shapes():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=32, embed_dim=EMBED_DIM)
    obs = torch.randn(4, OBS_DIM)
    prev_action = enc.initial_partner_action(4)
    rnn_state = enc.initial_state(4)
    embedding, new_rnn_state = enc.step(obs, prev_action, rnn_state)
    assert embedding.shape == (4, EMBED_DIM)
    assert new_rnn_state.shape == (4, 32)


def test_embedding_is_bounded():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=32, embed_dim=EMBED_DIM)
    obs = torch.randn(4, OBS_DIM) * 10
    prev_action = enc.initial_partner_action(4)
    rnn_state = enc.initial_state(4)
    embedding, _ = enc.step(obs, prev_action, rnn_state)
    assert torch.all(embedding >= -1.0) and torch.all(embedding <= 1.0)  # tanh output head


def test_encoder_is_deterministic_given_same_input():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=32, embed_dim=EMBED_DIM)
    enc.eval()
    obs = torch.randn(1, OBS_DIM)
    prev_action = enc.initial_partner_action(1)
    rnn_state = enc.initial_state(1)
    with torch.no_grad():
        e1, s1 = enc.step(obs, prev_action, rnn_state)
        e2, s2 = enc.step(obs, prev_action, rnn_state)
    assert torch.allclose(e1, e2)
    assert torch.allclose(s1, s2)


def test_encoder_processes_variable_length_sequences():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, hidden_size=32, embed_dim=EMBED_DIM)
    rnn_state = enc.initial_state(1)
    prev_action = enc.initial_partner_action(1)
    embeddings = []
    for t in range(10):
        obs = torch.randn(1, OBS_DIM)
        embedding, rnn_state = enc.step(obs, prev_action, rnn_state)
        prev_action = enc.action_to_onehot(t % NUM_ACTIONS)
        embeddings.append(embedding)
    assert len(embeddings) == 10
    # the embedding should actually change as more of the trajectory is observed
    assert not torch.allclose(embeddings[0], embeddings[-1])


def test_action_to_onehot():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS)
    onehot = enc.action_to_onehot(2)
    assert onehot.shape == (1, NUM_ACTIONS)
    assert onehot.sum().item() == 1.0
    assert onehot[0, 2].item() == 1.0


def test_no_partner_action_ablation_ignores_partner_actions():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, use_partner_actions=False)
    obs = torch.randn(2, OBS_DIM)
    state = enc.initial_state(2)
    a, _ = enc.step(obs, enc.action_to_onehot(np.array([0, 1])), state)
    b, _ = enc.step(obs, enc.action_to_onehot(np.array([2, 3])), state)
    assert torch.allclose(a, b)


def test_memoryless_ablation_depends_only_on_the_current_observation():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS, use_partner_actions=False, use_memory=False)
    obs = torch.randn(1, OBS_DIM)
    fresh, _ = enc.step(obs, enc.initial_partner_action(1), enc.initial_state(1))
    after_history, _ = enc.step(obs, enc.action_to_onehot(3), torch.randn(1, enc.hidden_size))
    assert torch.allclose(fresh, after_history)


def test_full_encoder_does_use_memory_and_partner_actions():
    enc = TrajectoryEncoder(OBS_DIM, NUM_ACTIONS)
    obs = torch.randn(1, OBS_DIM)
    base, _ = enc.step(obs, enc.action_to_onehot(0), enc.initial_state(1))
    other_action, _ = enc.step(obs, enc.action_to_onehot(3), enc.initial_state(1))
    other_memory, _ = enc.step(obs, enc.action_to_onehot(0), torch.randn(1, enc.hidden_size))
    assert not torch.allclose(base, other_action)
    assert not torch.allclose(base, other_memory)
