"""Unit tests for the gridworld environment (Section 6: "unit tests ...
to verify ... the shapes of observations, actions, and states being
correct")."""

import numpy as np
import pytest

from adhoc_teamwork.envs.gridworld_overcooked import (
    INTERACT,
    NUM_ACTIONS,
    OBS_DIM,
    OvercookedGridworld,
)


def test_reset_shapes():
    env = OvercookedGridworld(horizon=50)
    obs = env.reset(seed=0)
    assert obs.shape == (2, OBS_DIM)
    assert obs.dtype == np.float32


def test_obs_values_bounded():
    env = OvercookedGridworld(horizon=50)
    obs = env.reset(seed=0)
    assert np.all(obs >= -1e-6) and np.all(obs <= 1.0 + 1e-6)


def test_step_shapes_and_types():
    env = OvercookedGridworld(horizon=50)
    env.reset(seed=0)
    result = env.step([0, 0])
    assert result.obs.shape == (2, OBS_DIM)
    assert result.rewards.shape == (2,)
    assert isinstance(result.done, bool)
    # fully cooperative: both agents always receive the identical shared reward
    assert result.rewards[0] == result.rewards[1]


def test_episode_terminates_at_horizon():
    env = OvercookedGridworld(horizon=5)
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        result = env.step([4, 4])  # STAY
        done = result.done
        steps += 1
        assert steps <= 5
    assert steps == 5


def test_all_actions_are_legal_and_safe():
    env = OvercookedGridworld(horizon=50)
    env.reset(seed=0)
    for a0 in range(NUM_ACTIONS):
        for a1 in range(NUM_ACTIONS):
            env.reset(seed=1)
            result = env.step([a0, a1])  # should never raise
            assert result.obs.shape == (2, OBS_DIM)


def test_legal_actions_mask_shape():
    env = OvercookedGridworld(horizon=50)
    env.reset(seed=0)
    mask = env.legal_actions()
    assert mask.shape == (2, NUM_ACTIONS)
    assert mask.dtype == bool
    assert mask.all()


def test_full_cooperative_episode_delivers_soup():
    """A complementary onion-runner / pot-tender pair should actually be
    able to complete the onion -> cook -> dish -> serve chain at least
    once, exercising every interaction branch of the environment.
    """
    from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent

    env = OvercookedGridworld(horizon=150)
    onion = ScriptedBiasedAgent("onion_runner", noise=0.0)
    pot = ScriptedBiasedAgent("pot_tender", noise=0.0)
    onion.reset()
    pot.reset()
    obs = env.reset(seed=0)
    done = False
    while not done:
        a0, a1 = onion.act(obs[0]), pot.act(obs[1])
        result = env.step([a0, a1])
        obs = result.obs
        done = result.done
    assert env.num_delivered >= 1


def test_mismatched_roles_never_deliver():
    from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent

    env = OvercookedGridworld(horizon=150)
    a1 = ScriptedBiasedAgent("onion_runner", noise=0.0)
    a2 = ScriptedBiasedAgent("onion_runner", noise=0.0)
    a1.reset()
    a2.reset()
    obs = env.reset(seed=0)
    done = False
    while not done:
        result = env.step([a1.act(obs[0]), a2.act(obs[1])])
        obs = result.obs
        done = result.done
    assert env.num_delivered == 0


def test_reward_shaping_flag_adds_bonus():
    plain = OvercookedGridworld(horizon=1, reward_shaping=False)
    shaped = OvercookedGridworld(horizon=1, reward_shaping=True)
    for env in (plain, shaped):
        env.reset(seed=0)
        env.held = ["onion", None]
        env.positions = [(1, 2), env.positions[1]]
        env.directions = [1, env.directions[1]]  # facing DOWN, towards the pot
    plain_reward = plain.step([INTERACT, 4]).rewards[0]
    shaped_reward = shaped.step([INTERACT, 4]).rewards[0]
    assert shaped_reward > plain_reward
