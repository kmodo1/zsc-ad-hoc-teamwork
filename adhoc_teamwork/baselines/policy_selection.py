"""Policy-selection baseline: hard (argmax) analogue of the adaptive
ensemble, in the spirit of Barrett et al. (2017)'s "select an optimal
action from a set of pre-trained policies" and CBPR (Wang et al., 2024a)'s
"adaptively selects one [policy] based on the current partner". Instead of
continuously *blending* every sub-policy's action distribution (as
``AdaptiveEnsembleAgent`` does), this baseline commits each step to a
single sub-policy -- whichever one the (same, already-trained) partner
encoder + weighting network currently scores highest -- and acts only
according to that policy.

Reusing the trained encoder/weighting network (rather than training a
wholly separate selection mechanism) keeps the comparison to the adaptive
ensemble controlled: the only thing that differs is soft mixing vs. hard
selection of the same learned partner-compatibility signal.
"""

from __future__ import annotations

import torch

from adhoc_teamwork.ensemble.adaptive_ensemble_agent import TrajectoryConditionedAgent


class PolicySelectionAgent(TrajectoryConditionedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selected = None

    def reset(self, batch_size: int = 1) -> None:
        super().reset(batch_size)
        self.last_selected = None

    def _forward_probs(self, obs_t, available_t):
        weights, sub_probs = self._weights_and_sub_probs(obs_t, available_t)
        selected = weights.argmax(dim=-1)
        self.last_selected = selected.cpu().numpy()
        return sub_probs[torch.arange(obs_t.shape[0], device=obs_t.device), selected]
