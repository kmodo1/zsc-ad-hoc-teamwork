"""Unit tests for the PPO trainer (Section 6: "individual training steps
making valid changes to weights")."""

import copy

import torch

from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.training.ppo_trainer import (
    PPOConfig,
    collect_self_play_episode,
    compute_gae,
    ppo_update,
)
import numpy as np


def test_compute_gae_shapes_and_values():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.1, 0.1, 0.1, 0.0], dtype=np.float32)  # includes bootstrap
    dones = np.array([False, False, True])
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    # reward only arrives at the last step; earlier advantages should reflect that
    assert advantages[-1] > advantages[0]


def test_collect_self_play_episode_shapes():
    env = OvercookedGridworld(horizon=10)
    policy = ActorCritic(env.obs_dim, env.num_actions, hidden_size=8)
    buffers = collect_self_play_episode(env, policy, torch.device("cpu"))
    assert len(buffers) == 2
    for buf in buffers:
        assert len(buf) == 10
        assert buf.obs[0].shape == (env.obs_dim,)


def test_ppo_update_changes_parameters():
    env = OvercookedGridworld(horizon=10)
    cfg = PPOConfig(hidden_size=8, minibatch_size=8, n_epochs=2)
    policy = ActorCritic(env.obs_dim, env.num_actions, hidden_size=cfg.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    before = copy.deepcopy(list(policy.parameters()))

    buffers = []
    for _ in range(2):
        buffers.extend(collect_self_play_episode(env, policy, torch.device("cpu")))
    stats = ppo_update(policy, optimizer, buffers, cfg, torch.device("cpu"))

    after = list(policy.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "PPO update should change at least one parameter"
    assert stats["n_updates"] > 0


def test_ppo_update_stats_are_finite():
    env = OvercookedGridworld(horizon=10)
    cfg = PPOConfig(hidden_size=8, minibatch_size=8, n_epochs=1)
    policy = ActorCritic(env.obs_dim, env.num_actions, hidden_size=cfg.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    buffers = collect_self_play_episode(env, policy, torch.device("cpu"))
    stats = ppo_update(policy, optimizer, [buffers[0]], cfg, torch.device("cpu"))
    for v in stats.values():
        assert np.isfinite(v)
