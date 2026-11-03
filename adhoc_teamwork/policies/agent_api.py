"""Common inference-time interface shared by every kind of agent used in
evaluation: trained NN policies, scripted biased agents, the static
ensemble, the policy-selection baseline, and the adaptive ensemble agent.

Keeping this interface tiny is what lets the cross-play matrix, the
baselines, and the final evaluation harness mix and match any of these agent
types without caring which one they are.

Agents can run a whole batch of parallel episodes at once: ``reset(n)``,
then ``act_batch`` with arrays of shape ``(n, ...)``. ``act`` and
``action_probs`` are the single-episode case of the same thing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class Agent:
    """Abstract inference-time agent. Stateful across one episode, or one
    batch of parallel episodes."""

    def reset(self, batch_size: int = 1) -> None:
        """Called once at the start of an episode (or batch of episodes)."""
        raise NotImplementedError

    def act(self, obs: np.ndarray, deterministic: bool = False, available_actions: Optional[np.ndarray] = None) -> int:
        """Return a single discrete action given this agent's current observation."""
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray, available_actions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the agent's action probability distribution, shape (num_actions,).

        Used by the ensemble machinery to combine several agents' policies.
        Agents that cannot produce meaningful probabilities (pure scripted
        agents) may return a one-hot vector on the action they would take.
        """
        raise NotImplementedError

    def act_batch(self, obs: np.ndarray, deterministic: bool = False, available_actions: Optional[np.ndarray] = None) -> np.ndarray:
        """``obs``: (n, obs_dim). Returns actions, shape (n,).

        This default suits stateless agents; recurrent ones must override it.
        """
        masks = available_actions if available_actions is not None else [None] * len(obs)
        return np.array([self.act(o, deterministic, m) for o, m in zip(obs, masks)], dtype=np.int64)

    def observe_partner_action(self, partner_action) -> None:
        """Optional hook: called by the rollout loop right after each env
        step with the action(s) the *other* agent actually took that step
        (an int, or an array of shape (n,) in batch mode). Default no-op.
        Agents that track beliefs about their partner (the adaptive
        ensemble, the policy-selection baseline) override this.
        """
        pass


class TorchPolicyAgent(Agent):
    """Base class for agents backed by networks. Subclasses implement
    :meth:`_forward_probs`; everything else is derived from it.

    ``argmax_actions`` makes the agent always take its most probable action,
    which is how Section 4 of the proposal describes the ensemble's final
    action ("taking the most highly outputted action after applying weights").
    """

    def __init__(self, device: torch.device = torch.device("cpu"), argmax_actions: bool = False):
        self.device = device
        self.argmax_actions = argmax_actions

    def _forward_probs(self, obs_t: torch.Tensor, available_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Returns action probabilities (B, num_actions) and advances any
        recurrent state by one step."""
        raise NotImplementedError

    @torch.no_grad()
    def probs_batch(self, obs: np.ndarray, available_actions: Optional[np.ndarray] = None) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        available_t = None if available_actions is None else torch.as_tensor(available_actions, dtype=torch.bool, device=self.device)
        return self._forward_probs(obs_t, available_t)

    def act_batch(self, obs: np.ndarray, deterministic: bool = False, available_actions: Optional[np.ndarray] = None) -> np.ndarray:
        probs = self.probs_batch(obs, available_actions)
        if deterministic or self.argmax_actions:
            return probs.argmax(dim=-1).cpu().numpy()
        return torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

    def act(self, obs: np.ndarray, deterministic: bool = False, available_actions: Optional[np.ndarray] = None) -> int:
        batch_mask = None if available_actions is None else available_actions[None]
        return int(self.act_batch(obs[None], deterministic, batch_mask)[0])

    def action_probs(self, obs: np.ndarray, available_actions: Optional[np.ndarray] = None) -> np.ndarray:
        # NOTE: this advances the recurrent state (as does `act`).
        batch_mask = None if available_actions is None else available_actions[None]
        return self.probs_batch(obs[None], batch_mask)[0].cpu().numpy()
