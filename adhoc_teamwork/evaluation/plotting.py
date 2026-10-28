"""Plotting helpers (Section 7 deliverable: "plots showing performance and
policy-weight changes"). Kept dependency-light (matplotlib only) so these
run anywhere pytest does.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_method_comparison(results: Dict[str, List[float]], out_path: str, title: str = "Team return by method"):
    """``results``: {method_name: [per-episode returns]}."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    names = list(results.keys())
    means = [float(np.mean(results[n])) for n in names]
    stds = [float(np.std(results[n])) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 4))
    ax.bar(names, means, yerr=stds, capsize=4)
    ax.set_ylabel("mean team return")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_crossplay_matrix(names: List[str], matrix: np.ndarray, out_path: str, title: str = "Cross-play compatibility matrix"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.2 * len(names) + 2))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("agent-1 slot")
    ax.set_ylabel("agent-0 slot")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean team return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_trajectory(weight_history: np.ndarray, policy_names: List[str], out_path: str, title: str = "Ensemble weights over an episode"):
    """``weight_history``: (T, K) array of weights over time."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(policy_names):
        ax.plot(weight_history[:, k], label=name)
    ax.set_xlabel("timestep")
    ax.set_ylabel("ensemble weight")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
