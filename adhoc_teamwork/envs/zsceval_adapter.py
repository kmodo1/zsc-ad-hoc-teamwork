"""Adapter from the real ``ZSC-Eval`` / Overcooked-AI environment to our
:class:`TwoAgentEnv` interface.

Requires the ``zsceval`` conda environment (Python 3.9, ``gym==0.22``,
``numpy==1.22``, see ``ZSC-Eval/environment.yml``); use
``slurm/env.sh`` on the cluster, which also makes sure the *vendored*
``ZSC-Eval/`` is the copy that gets imported. The contract below is covered
by ``tests/test_zsceval_adapter.py``, and the end-to-end behaviour is checked
against ZSC-Eval's published benchmark numbers by
``scripts/validate_overcooked.py``.

Contract details that differ from what the raw env returns, and why:

* **Rewards.** The raw env's per-agent reward is
  ``sparse + reward_shaping_factor * shaped_i``, which is *not* a shared
  reward once shaping is on. ``StepResult.rewards`` passes that through (it
  is what a trainer wants), while ``StepResult.team_reward`` is always the
  sparse team reward (20 per soup), which is what every reported return uses.
* **Observations.** The raw env returns a lossless ``(H, W, C)`` grid scaled
  by 255. We flatten it and undo the scaling so our networks get
  sensibly-scaled inputs; :class:`~adhoc_teamwork.policies.zsceval_policy.ZSCEvalPolicy`
  restores the raw format for ZSC-Eval's own actors.
* **Actions.** The raw env expects ``[[a0], [a1]]``; we take ``[a0, a1]``.
* **Action masks.** ZSC-Eval policies were trained with
  ``use_available_actions=True``, so :meth:`legal_actions` must be passed to
  them at inference time or they do not behave as they did in training.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import numpy as np

from .base import StepResult, TwoAgentEnv

# Sparse reward for one delivered soup in the "old" Overcooked dynamics.
SOUP_DELIVERY_REWARD = 20.0


def build_overcooked_args(
    layout_name: str,
    horizon: int = 400,
    reward_shaping_factor: float = 0.0,
    random_index: bool = False,
) -> argparse.Namespace:
    """Build the ``all_args`` namespace ``Overcooked.__init__`` expects, from
    ZSC-Eval's own argument parsers (so defaults stay whatever it defines).
    """
    from zsceval.config import get_config
    from zsceval.overcooked_config import OLD_LAYOUTS, get_overcooked_args

    if layout_name not in OLD_LAYOUTS:
        raise ValueError(
            f"layout {layout_name!r} needs ZSC-Eval's 'overcooked_new' env, which this "
            f"adapter does not wrap. Supported layouts: {sorted(OLD_LAYOUTS)}"
        )
    parser = get_overcooked_args(get_config())
    parser.add_argument("--use_phi", default=False, action="store_true")
    argv = [
        "--env_name", "Overcooked",
        "--algorithm_name", "mappo",
        "--experiment_name", "adhoc_teamwork",
        "--layout_name", layout_name,
        "--num_agents", "2",
        "--episode_length", str(horizon),
        "--reward_shaping_factor", str(reward_shaping_factor),
        "--initial_reward_shaping_factor", str(reward_shaping_factor),
    ]  # fmt: skip
    if random_index:
        argv.append("--random_index")
    args = parser.parse_args(argv)
    args.old_dynamics = True
    return args


class ZSCEvalAdapter(TwoAgentEnv):
    def __init__(self, raw_env):
        """``raw_env``: an already-constructed
        ``zsceval.envs.overcooked.Overcooked_Env.Overcooked`` instance.
        """
        self.raw_env = raw_env
        self.obs_shape = tuple(raw_env.observation_space[0].shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.num_actions = int(raw_env.action_space[0].n)
        self.horizon = int(raw_env.episode_length)
        self.t = 0
        self.num_delivered = 0
        self._last_available_actions: Optional[np.ndarray] = None

    def _format_obs(self, both_agents_ob: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(o, dtype=np.float32).reshape(-1) / 255.0 for o in both_agents_ob], axis=0)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            # NOTE: this reseeds the *global* torch/numpy/random RNGs (see
            # zsceval.utils.train_util.setup_seed). The env itself is
            # deterministic given actions, so this only affects agents' sampling.
            self.raw_env.seed(seed)
        both_agents_ob, _share_obs, available_actions = self.raw_env.reset()
        self._last_available_actions = np.asarray(available_actions, dtype=bool)
        self.t = 0
        self.num_delivered = 0
        return self._format_obs(both_agents_ob)

    def step(self, actions: List[int]) -> StepResult:
        raw_actions = [[int(a)] for a in actions]
        both_agents_ob, _share_obs, reward, done, info, available_actions = self.raw_env.step(raw_actions)
        self._last_available_actions = np.asarray(available_actions, dtype=bool)
        team_reward = float(sum(info["sparse_r_by_agent"]))
        self.t += 1
        self.num_delivered += int(round(team_reward / SOUP_DELIVERY_REWARD))
        return StepResult(
            obs=self._format_obs(both_agents_ob),
            rewards=np.array([float(reward[0][0]), float(reward[1][0])], dtype=np.float32),
            done=bool(done[0]),
            info=info,
            team_reward=team_reward,
        )

    def legal_actions(self) -> np.ndarray:
        if self._last_available_actions is None:
            return np.ones((self.num_agents, self.num_actions), dtype=bool)
        return self._last_available_actions


class _NoShareObsOvercooked:
    """Mixin skipping the centralized-critic observation, which nothing in
    this package reads and which costs a second full state featurization on
    every step of every environment.
    """

    def _gen_share_observation(self, state):
        return None


def make_overcooked_env(
    layout_name: str,
    horizon: int = 400,
    reward_shaping_factor: float = 0.0,
    random_index: bool = False,
    evaluation: bool = True,
) -> ZSCEvalAdapter:
    """Construct the real Overcooked env behind our :class:`TwoAgentEnv`
    interface. The first call for a layout computes and caches its motion
    planner, which can take minutes; run ``scripts/validate_overcooked.py
    --prepare-only`` once per layout before launching parallel jobs.
    """
    from zsceval.envs.overcooked.Overcooked_Env import Overcooked

    class _Env(_NoShareObsOvercooked, Overcooked):
        pass

    args = build_overcooked_args(layout_name, horizon, reward_shaping_factor, random_index)
    # baselines_reproducible=False: otherwise construction calls np.random.seed(0).
    # stuck_time=0: "stuck" bookkeeping is only used by ZSC-Eval's own scripted agents.
    raw_env = _Env(args, run_dir=None, baselines_reproducible=False, featurize_type=("ppo", "ppo"), stuck_time=0, evaluation=evaluation)
    return ZSCEvalAdapter(raw_env)
