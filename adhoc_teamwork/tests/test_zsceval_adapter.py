"""Contract tests for the real-Overcooked adapter and the ZSC-Eval policy
wrapper. Skipped automatically outside the ``zsceval`` environment.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("zsceval")
pytest.importorskip("gym")

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState  # noqa: E402

from adhoc_teamwork.envs.vec_env import VecEnv  # noqa: E402
from adhoc_teamwork.envs.zsceval_adapter import SOUP_DELIVERY_REWARD, ZSCEvalAdapter, make_overcooked_env  # noqa: E402
from adhoc_teamwork.evaluation.rollout import run_episode, run_episodes_batched  # noqa: E402
from adhoc_teamwork.policies.nn_agent import NNAgent  # noqa: E402
from adhoc_teamwork.policies.networks import ActorCritic  # noqa: E402

LAYOUT = "unident_s"
STAY = Action.ACTION_TO_INDEX[Action.STAY]
INTERACT = Action.ACTION_TO_INDEX[Action.INTERACT]


@pytest.fixture(scope="module")
def env():
    return make_overcooked_env(LAYOUT, horizon=50)


def _place_player_holding_soup_at_serving(env: ZSCEvalAdapter) -> None:
    """Rewrite the env state so player 0 holds a finished soup, facing a
    serving counter, and is one INTERACT away from a delivery."""
    env.reset()
    mdp = env.raw_env.base_mdp
    state = env.raw_env.base_env.state
    other_pos = state.players[1].position
    for serve_pos in mdp.get_serving_locations():
        for direction in Direction.ALL_DIRECTIONS:
            pos = (serve_pos[0] - direction[0], serve_pos[1] - direction[1])
            if pos in mdp.get_valid_player_positions() and pos != other_pos:
                soup = ObjectState("soup", pos, ("onion", 3, 20))
                players = (PlayerState(pos, direction, soup), state.players[1])
                env.raw_env.base_env.state = OvercookedState(players, {}, order_list=None)
                return
    raise AssertionError("no reachable serving position found")


def test_reset_returns_flat_scaled_observations(env):
    obs = env.reset()
    assert obs.shape == (2, env.obs_dim)
    assert env.obs_dim == int(np.prod(env.obs_shape))
    # Lossless encoding is small counts; 255-scaled values would mean the
    # scaling was not undone.
    assert obs.max() <= 30.0 and obs.min() >= 0.0


def test_step_accepts_plain_int_actions_and_returns_contract(env):
    env.reset()
    result = env.step([STAY, STAY])
    assert result.obs.shape == (2, env.obs_dim)
    assert result.rewards.shape == (2,)
    assert result.done is False
    assert result.team_reward == 0.0
    assert env.t == 1


def test_episode_ends_at_horizon(env):
    env.reset()
    for t in range(env.horizon):
        result = env.step([STAY, STAY])
        assert result.done == (t == env.horizon - 1)
    assert env.t == env.horizon


def test_slots_map_to_players_in_order(env):
    """Slot 0's action must move player 0, and slot 0's observation must be
    from player 0's perspective."""
    env.reset()
    start = [p.position for p in env.raw_env.base_env.state.players]
    moved = False
    for action_index, direction in enumerate(Direction.ALL_DIRECTIONS):
        env.reset()
        env.step([action_index, STAY])
        now = [p.position for p in env.raw_env.base_env.state.players]
        assert now[1] == start[1], "slot 0's action moved player 1"
        moved = moved or now[0] != start[0]
    assert moved, "slot 0's action never moved player 0"

    obs = env.reset()
    raw_ob0, raw_ob1 = env.raw_env.featurize_fn(env.raw_env.base_env.state)
    assert np.allclose(obs[0], raw_ob0.reshape(-1) / 255.0)
    assert np.allclose(obs[1], raw_ob1.reshape(-1) / 255.0)
    assert not np.allclose(obs[0], obs[1]), "the two perspectives should differ"


def test_delivery_gives_team_reward_and_is_counted(env):
    _place_player_holding_soup_at_serving(env)
    result = env.step([INTERACT, STAY])
    assert result.team_reward == SOUP_DELIVERY_REWARD
    assert env.num_delivered == 1
    assert result.rewards[0] == result.rewards[1] == SOUP_DELIVERY_REWARD


def test_team_reward_stays_sparse_under_reward_shaping():
    """With shaping on, the per-agent training signal includes dense reward
    but the reported team reward must not."""
    shaped = make_overcooked_env(LAYOUT, horizon=50, reward_shaping_factor=1.0)
    mdp = shaped.raw_env.base_mdp
    shaped.reset()
    state = shaped.raw_env.base_env.state
    # Hold an onion facing a pot: placing it is a shaped event for that player.
    other_pos = state.players[1].position
    for pot_pos in mdp.get_pot_locations():
        for direction in Direction.ALL_DIRECTIONS:
            pos = (pot_pos[0] - direction[0], pot_pos[1] - direction[1])
            if pos in mdp.get_valid_player_positions() and pos != other_pos:
                onion = ObjectState("onion", pos)
                players = (PlayerState(pos, direction, onion), state.players[1])
                shaped.raw_env.base_env.state = OvercookedState(players, {}, order_list=None)
                result = shaped.step([INTERACT, STAY])
                assert result.team_reward == 0.0
                assert result.rewards[0] > 0.0, "placing an onion in a pot should earn shaped reward"
                assert result.rewards[1] == 0.0, "shaped reward is per-agent"
                return
    raise AssertionError("no reachable pot position found")


def test_legal_actions_mask_shape_and_interact_rule(env):
    env.reset()
    legal = env.legal_actions()
    assert legal.shape == (2, env.num_actions)
    assert legal.dtype == bool
    assert legal[:, STAY].all(), "STAY must always be legal"
    _place_player_holding_soup_at_serving(env)
    env.step([STAY, STAY])
    assert env.legal_actions()[0, INTERACT], "serving a held soup must be legal"


def test_masked_actions_are_never_sampled(env):
    policy = ActorCritic(env.obs_dim, env.num_actions, hidden_size=8)
    agent = NNAgent(policy)
    agent.reset()
    obs = env.reset()
    mask = np.array([True, False, False, False, False, True])
    probs = agent.action_probs(obs[0], available_actions=mask)
    assert probs[~mask].sum() < 1e-6
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_run_episode_works_end_to_end_on_the_real_env(env):
    policy = ActorCritic(env.obs_dim, env.num_actions, hidden_size=8)
    traj = run_episode(env, [NNAgent(policy), NNAgent(policy)], seed=0)
    assert traj.length == env.horizon
    assert traj.team_return == traj.num_delivered * SOUP_DELIVERY_REWARD


def test_batched_rollout_on_the_real_env():
    vec_env = VecEnv(lambda: make_overcooked_env(LAYOUT, horizon=10), 3)
    policy = ActorCritic(vec_env.obs_dim, vec_env.num_actions, hidden_size=8)
    result = run_episodes_batched(vec_env, [NNAgent(policy), NNAgent(policy)], seed=0)
    assert result.returns.shape == (3,)
    assert result.delivered.shape == (3,)


class TestZSCEvalPolicy:
    """Requires a local policy pool; skipped if there is none."""

    @pytest.fixture(scope="class")
    def pool_root(self):
        import os

        from adhoc_teamwork.population.zsceval_pool import default_pool_root

        root = default_pool_root()
        if not os.path.isdir(os.path.join(root, LAYOUT)):
            pytest.skip(f"no ZSC-Eval policy pool at {root}")
        return root

    def _raw_actor_probs(self, wrapper, raw_obs, rnn_state, mask):
        """What R_Actor itself computes, going through its own forward path."""
        actor = wrapper.actor
        not_done = torch.ones(raw_obs.shape[0], 1)
        features = actor.base(raw_obs)
        if wrapper.is_recurrent:
            features, rnn_state = actor.rnn(features, rnn_state, not_done)
        return actor.act.action_out(features, mask).probs, rnn_state

    @pytest.mark.parametrize(
        "rel_path,recurrent",
        [
            (f"{LAYOUT}/hsp/s1/hsp/hsp21_final_w0_actor.pt", False),
            (f"{LAYOUT}/traj/s2/traj-S2-s36/1.pt", True),
            (f"{LAYOUT}/fcp/s1/sp/sp1_final_actor.pt", None),
        ],
    )
    def test_matches_raw_actor_on_real_observations(self, env, pool_root, rel_path, recurrent):
        from adhoc_teamwork.policies.zsceval_policy import load_zsceval_policy

        wrapper = load_zsceval_policy(pool_root, LAYOUT, rel_path)
        if recurrent is not None:
            assert wrapper.is_recurrent == recurrent

        obs = env.reset()
        raw = env.raw_env.featurize_fn(env.raw_env.base_env.state)
        raw_obs = torch.as_tensor(np.stack(raw), dtype=torch.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mask = torch.as_tensor(env.legal_actions(), dtype=torch.bool)

        state = wrapper.initial_state(2)
        with torch.no_grad():
            ours, new_state = wrapper.action_probs(obs_t, state, mask)
            theirs, _ = self._raw_actor_probs(wrapper, raw_obs, state, mask)
        assert torch.allclose(ours, theirs, atol=1e-5)
        assert torch.allclose(ours.sum(-1), torch.ones(2))
        if wrapper.is_recurrent:
            assert not torch.allclose(new_state, state)

    def test_batched_and_single_inference_agree(self, env, pool_root):
        from adhoc_teamwork.policies.zsceval_policy import load_zsceval_policy

        wrapper = load_zsceval_policy(pool_root, LAYOUT, f"{LAYOUT}/traj/s2/traj-S2-s36/1.pt")
        obs = env.reset()
        legal = env.legal_actions()
        batched = NNAgent(wrapper)
        batched.reset(batch_size=2)
        single = [NNAgent(wrapper), NNAgent(wrapper)]
        for a in single:
            a.reset()
        for _ in range(5):
            batch_probs = batched.probs_batch(obs, legal).numpy()
            for i in range(2):
                assert np.allclose(batch_probs[i], single[i].action_probs(obs[i], legal[i]), atol=1e-5)
            result = env.step([STAY, STAY])
            obs, legal = result.obs, env.legal_actions()
