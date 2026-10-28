"""Common inference-time interface shared by every kind of agent used in
evaluation: trained NN policies, scripted biased agents, the static
ensemble, the policy-selection baseline, and the adaptive ensemble agent.

Keeping this interface tiny (reset / act / action_probs) is what lets the
cross-play matrix, the baselines, and the final evaluation harness mix and
match any of these agent types without caring which one they are.
"""

from __future__ import annotations

import numpy as np


class Agent:
    """Abstract inference-time agent. Stateful across one episode."""

    def reset(self) -> None:
        """Called once at the start of an episode."""
        raise NotImplementedError

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Return a single discrete action given this agent's current observation."""
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return the agent's action probability distribution, shape (num_actions,).

        Used by the ensemble machinery to combine several agents' policies.
        Agents that cannot produce meaningful probabilities (pure scripted
        agents) may return a one-hot vector on the action they would take.
        """
        raise NotImplementedError

    def observe_partner_action(self, partner_action: int) -> None:
        """Optional hook: called by the rollout loop right after each env
        step with the action the *other* agent actually took that step.
        Default no-op. Agents that track beliefs about their partner (the
        adaptive ensemble, the policy-selection baseline) override this.
        """
        pass
