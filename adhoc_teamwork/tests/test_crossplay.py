"""Unit tests for the cross-play evaluation matrix builder."""

import numpy as np

from adhoc_teamwork.crossplay.evaluate_crossplay import build_crossplay_matrix, self_play_vs_cross_play_gap
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.policies.networks import ActorCritic
from adhoc_teamwork.population.policy_pool import PolicyPool


def _tiny_pool(n=3):
    pool = PolicyPool()
    for i in range(n):
        pool.add(f"policy_{i}", ActorCritic(24, 6, hidden_size=8))
    return pool


def test_crossplay_matrix_shape():
    env = OvercookedGridworld(horizon=10)
    pool = _tiny_pool(3)
    result = build_crossplay_matrix(env, pool, n_episodes=1)
    assert result["matrix"].shape == (3, 3)
    assert result["delivered"].shape == (3, 3)
    assert result["names"] == ["policy_0", "policy_1", "policy_2"]


def test_crossplay_matrix_finite():
    env = OvercookedGridworld(horizon=10)
    pool = _tiny_pool(2)
    result = build_crossplay_matrix(env, pool, n_episodes=1)
    assert np.all(np.isfinite(result["matrix"]))


def test_self_play_vs_cross_play_gap_keys():
    env = OvercookedGridworld(horizon=10)
    pool = _tiny_pool(2)
    result = build_crossplay_matrix(env, pool, n_episodes=1)
    gap = self_play_vs_cross_play_gap(result)
    assert set(gap.keys()) == {"self_play_mean", "cross_play_mean", "gap"}
