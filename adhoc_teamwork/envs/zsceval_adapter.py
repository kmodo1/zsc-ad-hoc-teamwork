"""Adapter from the real ``ZSC-Eval`` / Overcooked-AI environment to our
:class:`TwoAgentEnv` interface, for running the method in this repository
against the real environment on the UMIACS cluster once a GPU + the
``zsceval`` conda environment (Python 3.9, ``gym==0.22``, ``numpy==1.22``,
see ``ZSC-Eval/environment.yml``) are available.

NOTE: this module cannot be imported or exercised in this repository's own
test suite -- this machine runs Python 3.13 with ``numpy>=2`` and no
``gym``, which the vendored ``ZSC-Eval`` package requires. It is included
so the rest of this codebase (population training, cross-play, the
trajectory encoder, the adaptive ensemble, baselines, evaluation) can be
pointed at the real Overcooked environment with no changes anywhere else,
simply by swapping ``OvercookedGridworld(...)`` for
``ZSCEvalAdapter(...)``. It has been written against, but not executed
against, ``zsceval.envs.overcooked.Overcooked_Env.Overcooked`` (see that
module for the exact ``reset``/``step`` contract this wraps); treat it as
a documented starting point to validate on the cluster, not a verified
implementation.

Usage on the cluster (sketch -- see ``ZSC-Eval/README.md`` for how to
build the ``all_args`` namespace and layout the policy config that
``Overcooked.__init__`` expects)::

    from zsceval.envs.overcooked.Overcooked_Env import Overcooked
    raw_env = Overcooked(all_args, run_dir=..., featurize_type=("ppo", "ppo"))
    env = ZSCEvalAdapter(raw_env)
    # from here on, `env` is a drop-in TwoAgentEnv for every other module
    # in this repository (population/, crossplay/, encoder/, ensemble/,
    # baselines/, evaluation/).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import StepResult, TwoAgentEnv


class ZSCEvalAdapter(TwoAgentEnv):
    def __init__(self, raw_env):
        """``raw_env``: an already-constructed
        ``zsceval.envs.overcooked.Overcooked_Env.Overcooked`` instance.
        """
        self.raw_env = raw_env
        self.obs_dim = int(np.prod(raw_env.observation_space[0].shape))
        self.num_actions = raw_env.action_space[0].n

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None and hasattr(self.raw_env, "seed"):
            self.raw_env.seed(seed)
        both_agents_ob, _share_obs, available_actions = self.raw_env.reset()
        self._last_available_actions = available_actions
        return np.stack([np.asarray(o, dtype=np.float32).reshape(-1) for o in both_agents_ob], axis=0)

    def step(self, actions: List[int]) -> StepResult:
        both_agents_ob, _share_obs, reward, done, info, available_actions = self.raw_env.step(tuple(actions))
        self._last_available_actions = available_actions
        obs = np.stack([np.asarray(o, dtype=np.float32).reshape(-1) for o in both_agents_ob], axis=0)
        # `reward` is [[r0], [r1]]; in the default fully-cooperative sparse
        # reward setting these are identical team rewards (as required by
        # our fully-cooperative TwoAgentEnv contract). If a ZSC-Eval config
        # with asymmetric/hidden per-agent reward (e.g. the HSP baseline)
        # is used, revisit this assumption.
        rewards = np.array([float(reward[0][0]), float(reward[1][0])], dtype=np.float32)
        is_done = bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)
        return StepResult(obs=obs, rewards=rewards, done=is_done, info=info)

    def legal_actions(self) -> np.ndarray:
        available_actions = getattr(self, "_last_available_actions", None)
        if available_actions is None:
            return np.ones((self.num_agents, self.num_actions), dtype=bool)
        return np.asarray(available_actions, dtype=bool)
