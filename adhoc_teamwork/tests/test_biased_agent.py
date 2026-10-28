"""Unit tests for the scripted biased agents used as unseen partners."""

import numpy as np
import pytest

from adhoc_teamwork.envs.gridworld_overcooked import NUM_ACTIONS, OvercookedGridworld
from adhoc_teamwork.evaluation.rollout import evaluate_pair
from adhoc_teamwork.policies.biased_agent import ROLES, ScriptedBiasedAgent


@pytest.mark.parametrize("role", ROLES)
def test_action_probs_valid_distribution(role):
    agent = ScriptedBiasedAgent(role, noise=0.1)
    agent.reset()
    env = OvercookedGridworld(horizon=10)
    obs = env.reset(seed=0)
    probs = agent.action_probs(obs[0])
    assert probs.shape == (NUM_ACTIONS,)
    assert np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


@pytest.mark.parametrize("role", ROLES)
def test_act_returns_legal_action(role):
    agent = ScriptedBiasedAgent(role, noise=0.1)
    agent.reset()
    env = OvercookedGridworld(horizon=10)
    obs = env.reset(seed=0)
    for _ in range(10):
        a = agent.act(obs[0])
        assert 0 <= a < NUM_ACTIONS
        obs = env.step([a, 4]).obs


def test_complementary_roles_cooperate_better_than_matched_roles():
    """This is the convention-mismatch phenomenon the whole project is
    about (Carroll et al., 2019): two agents biased towards the *same*
    role should do strictly worse than two agents biased towards
    *complementary* roles.
    """
    env = OvercookedGridworld(horizon=150)
    complementary = evaluate_pair(
        env, [ScriptedBiasedAgent("onion_runner", noise=0.0), ScriptedBiasedAgent("pot_tender", noise=0.0)], n_episodes=2
    )
    matched = evaluate_pair(
        env, [ScriptedBiasedAgent("onion_runner", noise=0.0), ScriptedBiasedAgent("onion_runner", noise=0.0)], n_episodes=2
    )
    assert complementary["mean_return"] > matched["mean_return"]
    assert complementary["mean_delivered"] > 0
    assert matched["mean_delivered"] == 0
