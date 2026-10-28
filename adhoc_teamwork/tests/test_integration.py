"""Integration test (Section 6: "Integration tests will verify the full
pipeline"): population training -> cross-play matrix -> adaptive-ensemble
training -> final evaluation against held-out partners -> ablation, all
wired together exactly as the CLI scripts in ``scripts/`` do, just at a
tiny scale so it runs in a few seconds.
"""

from __future__ import annotations

import numpy as np

from adhoc_teamwork.crossplay.evaluate_crossplay import build_crossplay_matrix
from adhoc_teamwork.envs.gridworld_overcooked import OvercookedGridworld
from adhoc_teamwork.evaluation.ablation import run_ablation
from adhoc_teamwork.evaluation.run_evaluation import evaluate_methods_against_partners
from adhoc_teamwork.policies.biased_agent import ScriptedBiasedAgent
from adhoc_teamwork.population.policy_pool import train_population
from adhoc_teamwork.training.ppo_trainer import PPOConfig
from adhoc_teamwork.training.train_adaptive_ensemble import EnsembleTrainConfig, train_adaptive_ensemble


def test_full_pipeline_runs_end_to_end():
    env = OvercookedGridworld(horizon=20)
    cfg = PPOConfig(hidden_size=8, minibatch_size=16, n_epochs=1)

    # 1) population of diverse self-play policies
    pool = train_population(env, n_policies=2, n_iterations=2, episodes_per_iteration=1, cfg=cfg, base_seed=0)
    assert len(pool) == 2

    # 2) cross-play compatibility matrix
    crossplay = build_crossplay_matrix(env, pool, n_episodes=1)
    assert crossplay["matrix"].shape == (2, 2)
    assert np.all(np.isfinite(crossplay["matrix"]))

    # 3) adaptive ensemble: partner trajectory encoder + weighting network
    sub_policies = [pool[name] for name in pool.names()]

    def partner_sampler():
        return ScriptedBiasedAgent("onion_runner", noise=0.2)

    encoder, weighting_net, train_stats = train_adaptive_ensemble(
        env,
        sub_policies,
        partner_sampler,
        embed_dim=8,
        encoder_hidden=16,
        weighting_hidden=16,
        n_iterations=2,
        episodes_per_iteration=1,
        cfg=EnsembleTrainConfig(),
        seed=0,
    )
    assert len(train_stats.returns) == 2

    # 4) final evaluation vs. held-out partners (methods: self-play, static
    #    ensemble, policy-selection, adaptive ensemble)
    held_out_partners = [ScriptedBiasedAgent("pot_tender", noise=0.0), ScriptedBiasedAgent("random_noise")]
    results = evaluate_methods_against_partners(
        env, sub_policies, encoder, weighting_net, held_out_partners, self_play_policy=sub_policies[0], n_episodes_per_partner=1
    )
    assert set(results.keys()) == {"self_play", "static_ensemble", "policy_selection", "adaptive_ensemble"}
    for name, returns in results.items():
        assert len(returns) == len(held_out_partners)
        assert all(np.isfinite(r) for r in returns)

    # 5) ablation: adaptive vs. uniform weighting on the same sub-policies
    ablation_results = run_ablation(env, sub_policies, encoder, weighting_net, held_out_partners, n_episodes_per_partner=1)
    assert set(ablation_results.keys()) == {"adaptive_weighting", "uniform_weighting"}
    for returns in ablation_results.values():
        assert len(returns) == len(held_out_partners)
