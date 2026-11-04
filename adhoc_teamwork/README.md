# Adaptive Ensemble Zero-Shot Coordination (this package)

Implementation of the project proposal ("Adaptive Ensemble Weighting for
Zero-Shot Ad-Hoc Teamwork", Powell / Patel / Jain, 18 Sept 2026): instead of
training one fixed zero-shot-coordination policy, we train a **diverse
population** of coordination policies, a **GRU-based partner trajectory
encoder**, and a **weighting network** that continuously re-weights the
population into a single ensemble policy based on the observed behavior of
the current (possibly unseen) partner.

This package (`adhoc_teamwork/`) lives alongside the vendored `ZSC-Eval/`
benchmark the proposal builds on. The method runs on two environments: a
small dependency-free gridworld, for quick checks, and the real Overcooked env
from `ZSC-Eval`, for the experiments. See **Running on real Overcooked** and
**Results** below.

## Repository layout

```
adhoc_teamwork/
  envs/         gridworld_overcooked.py  -- the dependency-light cooperative gridworld
                zsceval_adapter.py       -- adapter to the real ZSC-Eval / Overcooked env
                vec_env.py               -- steps parallel env copies in lock-step
  policies/     networks.py     -- recurrent actor-critic
                agent_api.py    -- common inference-time Agent interface
                nn_agent.py     -- wraps a trained policy for inference
                zsceval_policy.py -- makes a trained ZSC-Eval actor usable as a policy
                biased_agent.py -- scripted "unseen partner" agents
  training/     ppo_trainer.py             -- self-play PPO (produces the population)
                train_adaptive_ensemble.py -- trains encoder+weighting net (batched, resumable)
  population/   policy_pool.py   -- trains / saves / loads the gridworld population
                zsceval_pool.py  -- named policy sets from a ZSC-Eval policy pool
  experiment.py  which policies play which role on real Overcooked
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
                benchmark.py        -- resumable method x partner x position results table
                aggregate.py        -- method families, oracle references, paired bootstrap
                plotting.py         -- matplotlib helpers
  scripts/      CLI entry points (see Quickstart and Running on real Overcooked)
  slurm/        env.sh, run.sbatch, submit_overcooked.sh for the UMIACS cluster
  configs/      demo.yaml (gridworld demo in results/) and cluster.yaml (real Overcooked)
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
88 unit + integration tests (25 of them need the `zsceval` environment and
skip themselves without it, and some of those also need a policy pool) cover:
the real-Overcooked adapter's contract and the ZSC-Eval policy wrapper;
that evaluation partners are disjoint from training partners and sub-policies;
resumable training and evaluation; environment observation/action/state
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

## Running on real Overcooked

Uses the vendored `ZSC-Eval/` through `envs/zsceval_adapter.py`, in the `zsceval`
conda environment (Python 3.9, `gym==0.22`, `numpy==1.22`; see
`ZSC-Eval/environment.yml`). Everything else in this package only depends on
the `TwoAgentEnv` interface.

Policies come from a ZSC-Eval policy pool, which is not in this repository:

```bash
git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool
export ZSC_POLICY_POOL=$PWD/policy_pool
```

One-time setup on the cluster, then submit from the repository root:

```bash
export ADHOC_SCRATCH=/cmlscratch/$USER/adhoc_teamwork
mkdir -p $ADHOC_SCRATCH/logs
/path/to/envs/zsceval/bin/python -m venv --system-site-packages $ADHOC_SCRATCH/venv
$ADHOC_SCRATCH/venv/bin/pip install pytest

# Build each layout's cached motion planner once, before any parallel jobs.
sbatch -J adhoc-prepare --gres=gpu:0 adhoc_teamwork/slurm/run.sbatch \
    python -m adhoc_teamwork.scripts.validate_overcooked --layouts unident_s random0_medium --prepare-only

# Check the adapter against ZSC-Eval's published benchmark numbers.
sbatch -J adhoc-validate adhoc_teamwork/slurm/run.sbatch \
    python -m adhoc_teamwork.scripts.validate_overcooked --layouts unident_s --out $ADHOC_SCRATCH/runs/validation

# Evaluation jobs, plus a train-then-evaluate chain per seed. See the script
# header for options, e.g. ARMS="actor_critic:sp actor_critic:sp:memoryless".
bash adhoc_teamwork/slurm/submit_overcooked.sh

# Tables and plots. Safe to run while jobs are still going.
python -m adhoc_teamwork.scripts.analyze_overcooked --root $ADHOC_SCRATCH/runs --layout random0_medium
```

Scavenger jobs get preempted and requeued, so training and evaluation resume
from their last checkpoint or finished cell. Note that `slurm/env.sh` puts the
vendored `ZSC-Eval/` first on `PYTHONPATH`, because a conda environment may have
a different checkout pip-installed.

**Watching runs live.** Logs are in `$ADHOC_SCRATCH/logs/`. To mirror progress to
Weights & Biases, put your own settings in `$ADHOC_SCRATCH/local.env` (it is
outside the repository, so each person's stay separate):

```bash
export WANDB_ENTITY=<your entity>
export WANDB_PROJECT=<your project>
```

`submit_overcooked.sh` then also starts `scripts/sync_wandb.py`. It only reads
files the jobs already write, so training has no dependency on wandb.

## Results

Everything below is on real Overcooked, against ZSC-Eval's 20 held-out biased
evaluation partners (10 distinct agents, at two checkpoints each), with 20
episodes in each player position. Returns are sparse team reward, 20 per soup.
Confidence intervals are 95% paired bootstraps over the 10 distinct agents, with
no correction for multiple comparisons. Files are in `results/overcooked/`.

The demo in `results/` outside that folder is the small gridworld run, which
only shows that the pipeline runs.

**Validation.** Replaying ZSC-Eval's published pairings through our adapter,
wrappers and rollout code reproduces its numbers: mean return 258.8 vs. 258.5
published on `unident_s` (r = 0.997 over 240 pairings) and 119.4 vs. 119.5 on
`random0_medium` (r = 0.995 over 240 pairings).

**Main comparison.** The ensemble is built from ZSC-Eval's 10-member TrajeDi
population (`traj-s10`). The weighting network trained for 1000 iterations with
an actor-critic estimator, on 3 seeds, against self-play partners only: it never
saw a biased agent. Baselines have 5 seeds each.

| Method | `unident_s` | `random0_medium` |
|---|---:|---:|
| Adaptive ensemble, sampled actions | 175.5 | 143.5 |
| Adaptive ensemble, argmax action (as in Section 4) | 176.4 | 148.2 |
| Policy selection (hard, same learned signal) | 176.0 | 147.5 |
| Learned fixed weights (ablation) | 175.7 | 127.2 |
| Uniform weights (static ensemble) | 176.6 | 108.7 |
| Best baseline | 305.6 (MEP) | 134.3 (HSP) |
| TrajeDi / HSP / FCP / MEP | 298.9 / 290.5 / 304.7 / 305.6 | 103.8 / 134.3 / 118.3 / 116.3 |
| MAPPO (self-play) | 179.8 | 118.4 |
| Best sub-policy per partner, in hindsight | 184.6 | 150.0 |
| Best single sub-policy, in hindsight | 178.2 | 139.3 |

The argmax row compares an argmax ensemble with baselines that sample their
actions, as ZSC-Eval's benchmark does; the sampled row is the like-for-like one.

On `random0_medium` the adaptive ensemble (sampled) beats uniform weights by
+34.8 [+19.4, +51.5], learned fixed weights by +16.3 [+8.3, +25.0], and TrajeDi,
FCP, MEP and MAPPO by +25 to +40. It does **not** clearly beat HSP
(+9.2 [-10.0, +25.8]) or the best single sub-policy in hindsight
(+4.2 [-5.6, +15.1]). On `unident_s` it is indistinguishable from uniform weights
(-1.1 [-3.1, +0.1]) and far below the stage-2 baselines (-115 to -130).

Two findings shape how to read this:

- **The population limits the result.** On `unident_s`, picking the best TrajeDi
  member for each partner in hindsight would reach only 184.6, against about 300
  for the stage-2 baselines. The per-partner best is a reference and not a strict
  bound, since switching within an episode can beat it: against held-out self-play
  and MEP partners on `random0_medium`, the ensemble does (338-352 vs. 328-337).
- **The gain may not come from modeling the partner.** On `random0_medium`, weights
  vary about four times more within an episode than between partners
  (std 0.060 vs. 0.016 in training), and look broadly similar across partners
  (`results/overcooked/random0_medium/traj-s10/weight_traces_actor_critic-sp.png`).
  That is what gating on the current game state would look like, which answers a
  different question from the proposal's. The `memoryless` and `no_partner_action`
  encoder ablations test this directly.

**Candidate populations**, measured but with no weighting network trained on them yet:

| Layout | Population | Uniform | Best single, hindsight | Best per partner, hindsight | Best baseline |
|---|---|---:|---:|---:|---:|
| `unident_s` | `traj-s10` | 176.6 | 178.2 | 184.6 | 305.6 |
| | `s2-mix` (stage-2 agents) | 297.1 | 310.2 | 323.0 | |
| | `hsp-w1` (specialists) | 192.8 | 217.7 | 278.7 | |
| `random0_medium` | `traj-s10` | 108.7 | 139.3 | 150.0 | 134.3 |
| | `s2-mix` | 112.5 | 148.9 | 156.7 | |
| | `hsp-w1` | 97.2 | 115.4 | 154.6 | |

## Status against the proposal

Done: the adapter is validated, and the method, its baselines and ablations run
end to end on real Overcooked on two layouts. TrajeDi, HSP and MAPPO baselines are
ZSC-Eval's own trained agents, so they are at full fidelity.

Not done:

- **Population training.** All policies come from a pre-trained pool; none have been
  trained in this repository for real Overcooked. `scripts/train_population.py` and
  `training/ppo_trainer.py` are gridworld-only. ZSC-Eval's trainers would be the way
  to train new ones.
- **The choice of population**, which Section 9 reserves as a human judgment call.
- **The REINFORCE arm.** `--advantage reinforce` is implemented and tested but has not
  been run at scale on real Overcooked, so actor-critic has not been compared with it.
- **More layouts and seeds.** Only two layouts, three training seeds.
- **COLE and E3T baselines**, which ZSC-Eval's pool has but this evaluation leaves out.
- **Human play** against the ensemble through the Overcooked UI (Section 4), which
  needs an agent wrapper for `ZSC-Eval/zsceval/human_exp/`.
