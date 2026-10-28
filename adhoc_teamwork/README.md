# Adaptive Ensemble Zero-Shot Coordination (this package)

Implementation of the project proposal ("Adaptive Ensemble Weighting for
Zero-Shot Ad-Hoc Teamwork", Powell / Patel / Jain, 18 Sept 2026): instead of
training one fixed zero-shot-coordination policy, we train a **diverse
population** of coordination policies, a **GRU-based partner trajectory
encoder**, and a **weighting network** that continuously re-weights the
population into a single ensemble policy based on the observed behavior of
the current (possibly unseen) partner.

This package (`adhoc_teamwork/`) lives alongside the vendored `ZSC-Eval/`
benchmark the proposal builds on. See **Relationship to ZSC-Eval** below for
why the method is implemented here against a lightweight custom
environment rather than directly against `ZSC-Eval`.

## Repository layout

```
adhoc_teamwork/
  envs/         gridworld_overcooked.py  -- the dependency-light cooperative
                                            gridworld everything below runs on
                zsceval_adapter.py       -- adapter to the real ZSC-Eval /
                                            Overcooked env, for the cluster
  policies/     networks.py     -- recurrent actor-critic
                agent_api.py    -- common inference-time Agent interface
                nn_agent.py     -- wraps a trained ActorCritic for inference
                biased_agent.py -- scripted "unseen partner" agents
  training/     ppo_trainer.py             -- self-play PPO (produces the population)
                train_adaptive_ensemble.py -- REINFORCE training of encoder+weighting net
  population/   policy_pool.py  -- trains / saves / loads the population
  crossplay/    evaluate_crossplay.py -- NxN cross-play compatibility matrix
  encoder/      trajectory_encoder.py -- GRU partner trajectory encoder
  ensemble/     weighting_network.py       -- embedding -> per-policy weights
                adaptive_ensemble_agent.py -- the proposed method, as an Agent
                checkpoint.py              -- save/load helper
  baselines/    self_play.py       -- single trained policy, zero-shot
                static_ensemble.py -- same population, fixed uniform weights
                policy_selection.py-- hard (argmax) selection, Barrett-et-al./CBPR-style
  evaluation/   rollout.py          -- generic two-agent episode runner
                run_evaluation.py   -- compares all methods on held-out partners
                ablation.py         -- adaptive vs. uniform weighting, same sub-policies
                plotting.py         -- matplotlib helpers
  scripts/      CLI entry points (see Quickstart)
  configs/      demo.yaml (fast, used for results/) and cluster.yaml (real-scale)
  tests/        pytest unit + integration tests
```

## Quickstart (reproduction instructions)

```bash
pip install -r adhoc_teamwork/requirements.txt
cd zsc-ad-hoc-teamwork      # repo root (parent of adhoc_teamwork/)

# 1. Train the population of diverse self-play coordination policies.
python -m adhoc_teamwork.scripts.train_population --out adhoc_teamwork/results/population

# 2. Cross-play compatibility matrix: shows which population members work
#    well together and which don't (the self-play/cross-play gap).
python -m adhoc_teamwork.scripts.build_crossplay_matrix \
    --population adhoc_teamwork/results/population --out adhoc_teamwork/results/crossplay

# 3. Train the partner trajectory encoder + adaptive weighting network
#    (sub-policies frozen; trained against a restricted set of scripted partners).
python -m adhoc_teamwork.scripts.train_adaptive_ensemble \
    --population adhoc_teamwork/results/population --out adhoc_teamwork/results/ensemble

# 4. Final comparison against held-out (unseen) partners: self-play vs.
#    static ensemble vs. policy-selection vs. the adaptive ensemble.
python -m adhoc_teamwork.scripts.evaluate \
    --population adhoc_teamwork/results/population \
    --ensemble adhoc_teamwork/results/ensemble/ensemble.pt --out adhoc_teamwork/results/evaluation

# 5. Ablation: learned weighting vs. uniform weighting on the same sub-policies.
python -m adhoc_teamwork.scripts.run_ablation \
    --population adhoc_teamwork/results/population \
    --ensemble adhoc_teamwork/results/ensemble/ensemble.pt --out adhoc_teamwork/results/ablation
```

Each script takes `--help` for its full set of options; `configs/demo.yaml`
documents the settings used to produce `results/` in this repository, and
`configs/cluster.yaml` documents the (untuned) starting point for the
real, larger-scale runs on the UMIACS cluster.

### Tests

```bash
pip install pytest
pytest adhoc_teamwork/tests -q
```
43 unit + integration tests cover: environment observation/action/state
shapes (incl. that a complementary onion-runner / pot-tender pair can
actually complete the full delivery chain, and that two agents with the
*same* bias never do -- the convention-mismatch phenomenon motivating the
whole project); the actor-critic network's output distributions; the
trajectory encoder's embeddings; that a PPO update step actually changes
parameters; that sub-policies stay frozen while the ensemble trains; the
weighting network, static ensemble, policy-selection baseline and adaptive
ensemble; the cross-play matrix builder; and a full end-to-end pipeline run
(population -> cross-play -> ensemble training -> evaluation -> ablation).

## The environment

`envs/gridworld_overcooked.py` is a small (5x4), dependency-free,
Overcooked-*style* cooperative gridworld: fetch onions, cook them in a pot
(3 onions, 5-step cook time), serve on a plate, deliver. It keeps the one
property that actually matters for this project: the two agent roles are
**symmetric**, so independently self-play-trained policies can (and do, see
`crossplay/`) settle on different, mutually-incompatible conventions about
who does what -- exactly the Carroll et al. (2019) / Hu et al. (2020)
failure mode the proposal targets. It is not a substitute for the real
Overcooked environment; it exists so the full method can be implemented,
run, and unit-tested without `ZSC-Eval`'s pinned `gym==0.22` / `numpy==1.22`
/ Python-3.9 toolchain.

## Relationship to ZSC-Eval

The vendored `ZSC-Eval/` package requires Python 3.9, `numpy==1.22`, and
`gym==0.22` (see `ZSC-Eval/environment.yml`). This repository is being
developed on Python 3.13 with current `numpy`/`torch`, where `ZSC-Eval`
cannot be imported. Rather than block the whole method on provisioning that
legacy toolchain, the method (population training, cross-play, the
trajectory encoder, the adaptive ensemble, every baseline) is implemented
against the `TwoAgentEnv` interface in `envs/base.py`, with two
implementations:

- `envs/gridworld_overcooked.py` -- runs everywhere, including CI/this repo's tests.
- `envs/zsceval_adapter.py` -- wraps `zsceval.envs.overcooked.Overcooked_Env.Overcooked`
  for the real experiments. **This has been written against, but not run
  against, the real environment** (this machine cannot import `gym`/`zsceval`);
  validating and fixing it up is the first step once the `zsceval` conda
  environment is available (on the UMIACS cluster, per the proposal).

Swapping environments requires no changes anywhere else in this package --
every other module only depends on the `TwoAgentEnv` interface.

## Status vs. the proposal timeline

As of this writing the proposal's timeline (Section 8) has us at
"Oct 15-28: Implement static ensemble baselines", about to start
"Oct 29-Nov 11: Build the trajectory encoder and train the adaptive
weighting network". This package already implements the full pipeline
through "Nov 12-25: Integrate and evaluate the adaptive ensemble" -- i.e.
every piece of Section 3's goal list -- against the lightweight gridworld,
with `results/` populated by one concrete run (3 self-play policies, 250
PPO iterations each; see `results/*/results.json` and the corresponding
plots).

That run is a **pipeline-correctness demonstration, not a converged
research result** -- it is small and fast on purpose (runs in under 10
minutes on a laptop CPU). The cross-play matrix does show the expected
self-play/cross-play gap (`results/crossplay/self_play_vs_cross_play_gap.json`:
self-play mean return 1.67 vs. cross-play mean 0.0), and the ablation shows
the learned weighting network outperforming uniform weighting on the same
sub-policies (`results/ablation/results.json`: 5.67 vs. 5.00 mean return).
But in the final held-out-partner comparison
(`results/evaluation/results.json`) the adaptive ensemble does *not* yet
beat the simpler baselines (self-play 3.0, static ensemble 2.5,
policy-selection 2.5, adaptive ensemble 1.5, all with large standard
deviations relative to the mean) -- consistent with the population and the
weighting network both still being under-trained at this scale (the
`results/evaluation/weight_trajectory.png` weights are nearly flat across
an episode rather than visibly adapting). Resolving that -- more
population policies trained for longer, more adaptive-ensemble training
iterations, retuned entropy/learning-rate -- is exactly the "Nov 26-Dec 6:
Run ablation experiments, compare against baselines, analyze results"
timeline step, and is explicitly deferred to the cluster-scale run (see
`configs/cluster.yaml`), not something to over-fit away on a laptop.

What is explicitly **not** done yet, and is deferred to the cluster per
Section 4 ("If larger GPUs are required, some of our members have access
to the UMIACS cluster"):

- Validating `envs/zsceval_adapter.py` against the real Overcooked env and
  re-running the whole pipeline there.
- Training the population, and the encoder/weighting network, at a scale
  that actually converges well (`results/` here uses small, fast-running
  networks and a few hundred PPO iterations -- enough to demonstrate
  learning and a working, tested pipeline, not a converged result).
- Human play against the ensemble via the Overcooked UI (Section 4).
- The ablations and baseline comparisons called for in Section 4
  (Trajectory Diversity, Hidden-utility Self-Play, MAPPO) implemented at
  full fidelity -- `baselines/self_play.py` notes where our simplified
  self-play PPO trainer already covers the MAPPO-style comparison, and
  where it does not.
