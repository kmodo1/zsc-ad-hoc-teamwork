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

**The populations.** Each ensemble combines one of three populations, with a weighting
network trained for 1000 iterations with an actor-critic estimator, on 3 seeds.
Baselines have 5 seeds.

- `traj-full10`: a whole 10-member TrajeDi stage-1 population. Each member trained in
  self-play with a bonus for differing from the others, and never with anyone else.
- `s2-mix`: 12 stage-2 agents (3 seeds each of TrajeDi, HSP, FCP, MEP). These are the
  baselines themselves, each already trained to work with many partners.
- `hsp-w1`: 10 specialists, each the teammate co-trained with one biased agent outside
  the benchmark set.

Unless noted, `traj-full10` trained against self-play partners, and `s2-mix` and
`hsp-w1` against self-play partners plus biased agents outside the benchmark set. Those
biased agents are never evaluation partners, but they come from the same generator, so
that is a weaker test of generalization (see **Training partners** below).

Every method ran both ways: sampling its action, or taking the most probable one
(argmax, which is how Section 4 describes the ensemble). Compare within a column.

| Layout | Population | Uniform, sampled | Adaptive, sampled | Uniform, argmax | Adaptive, argmax |
|---|---|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | 175.9 | 174.5 | 171.2 | 176.7 |
| | `s2-mix` | 297.1 | 301.2 | 316.9 | **321.0** |
| | `hsp-w1` | 192.8 | 267.6 | 236.0 | 269.0 |
| | *Best baseline (MEP)* | | *305.6* | | *308.6* |
| `random0_medium` | `traj-full10` | 90.9 | 140.8 | 115.0 | 142.8 |
| | `s2-mix` | 112.5 | 140.6 | 156.4 | **162.3** |
| | `hsp-w1` | 97.2 | 142.8 | 115.2 | 152.7 |
| | *Best baseline (HSP)* | | *134.3* | | *140.8* |

**Adaptive against uniform weights, same population.** It helps in most cases, by an
amount that depends heavily on the population:

| Layout | Population | Sampled | Argmax |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | -1.3 [-4.4, +0.7] | +5.5 [-0.4, +13.7] |
| | `s2-mix` | +4.1 [+2.2, +5.9] | +4.1 [-2.6, +12.1] |
| | `hsp-w1` | +74.8 [+57.6, +94.9] | +33.0 [+6.5, +52.9] |
| `random0_medium` | `traj-full10` | +49.9 [+28.3, +73.9] | +27.7 [+5.7, +51.3] |
| | `s2-mix` | +28.1 [+22.4, +34.3] | +6.0 [+0.8, +11.4] |
| | `hsp-w1` | +45.6 [+21.3, +74.5] | +37.5 [+11.7, +68.6] |

**Adaptive against the best baseline, like for like.**

| Layout | Population | Sampled | Argmax |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | -131.0 [-183.5, -78.4] | -131.9 [-185.3, -78.4] |
| | `s2-mix` | -4.4 [-5.9, -3.0] | **+12.4 [+5.1, +20.7]** |
| | `hsp-w1` | -37.9 [-59.6, -19.8] | -39.6 [-64.5, -18.1] |
| `random0_medium` | `traj-full10` | +6.5 [-10.5, +21.3] | +2.0 [-13.8, +15.4] |
| | `s2-mix` | +6.3 [-1.1, +13.2] | **+21.6 [+11.1, +30.8]** |
| | `hsp-w1` | +8.5 [-4.5, +20.3] | +12.0 [-2.7, +25.4] |

Only `s2-mix` with argmax beats the best baseline on both layouts, and most of that does
not come from adaptation. The same ensemble with *uniform* weights and argmax already
gets +8.3 [-4.9, +20.5] and +15.6 [+0.9, +29.2], and with no memory it scores 317.7 and
160.8.

**Is the gain from observing the partner's trajectory?** The encoder ablations test this.
Each row is the complete encoder minus an ablated one, so a positive number means the
removed part helped:

| Layout | Population | No memory, sampled | No memory, argmax | Partner actions hidden, sampled | Partner actions hidden, argmax |
|---|---|---:|---:|---:|---:|
| `unident_s` | `hsp-w1` | +35.1 [+15.9, +55.3] | +2.3 [-24.6, +28.6] | +0.8 [-3.9, +6.3] | -2.2 [-12.5, +9.1] |
| | `s2-mix` | +2.8 [+1.4, +4.1] | +3.3 [-2.5, +10.0] | not run | not run |
| `random0_medium` | `hsp-w1` | +6.9 [+3.8, +10.2] | +7.8 [+3.4, +12.3] | -0.1 [-1.8, +1.2] | +0.2 [-1.6, +1.8] |
| | `s2-mix` | +6.9 [+4.1, +10.1] | +1.6 [-0.6, +4.0] | not run | not run |
| | `traj-full10` | +4.2 [+1.4, +7.2] | -1.6 [-7.6, +3.6] | +0.7 [-1.5, +3.5] | -2.7 [-3.9, -1.6] |

- **`hsp-w1` shows real partner adaptation.** Compare
  `results/overcooked/unident_s/hsp-w1/weight_traces_actor_critic-sp+biased.png` with
  the `-memoryless` version. With memory, the weights split by partner within about the
  first 50 steps, and the two checkpoints of one agent tend to go the same way. Without
  memory, all 20 panels look almost the same (std between partners 0.065 vs. 0.010 on
  `unident_s`). `s2-mix` and `traj-full10` show less of this on both layouts (0.009-0.035,
  against 0.047-0.065 for `hsp-w1`), which fits members not specialized to partner styles.
- **Memory helps, but is not most of the gain.** Where the gain over uniform weights is
  sizeable, the no-memory version keeps 53-92% of it with sampled actions and 73% or more
  with argmax. So much of the benefit is gating between sub-policies on the current
  state, which does include the partner's position and held item.
- **The partner's action input adds nothing measurable.** Hiding it makes no detectable
  difference in five of six comparisons and slightly helps in the sixth. This does not
  mean information about the partner is useless: weights still split by partner without
  it (std 0.064), since memory can infer behavior from the observations.

**Training partners.** The `hsp-w1` results above depend on the training partners
resembling the evaluation partners. Trained against self-play partners only, the gain on
`unident_s` mostly disappears, which is the risk Section 9 warns about. It holds up
better on `random0_medium`.

| Layout | Trained against | Adaptive, sampled | Against uniform | Adaptive, argmax | Against uniform |
|---|---|---:|---:|---:|---:|
| `unident_s` | Self-play only | 222.2 | +29.4 [-1.3, +62.7] | 216.2 | -19.9 [-46.3, +6.3] |
| | Self-play + other biased agents | 267.6 | +74.8 [+57.6, +94.9] | 269.0 | +33.0 [+6.5, +52.9] |
| `random0_medium` | Self-play only | 138.5 | +41.3 [+13.5, +74.7] | 142.9 | +27.7 [+0.0, +60.3] |
| | Self-play + other biased agents | 142.8 | +45.6 [+21.3, +74.5] | 152.7 | +37.5 [+11.7, +68.6] |

**Cross-play.** Mean return when a member plays itself against when it plays another
member, over all ordered pairs with 20 episodes each (`crossplay_matrix.png` in each
population's folder):

| Layout | Population | Self-play | Cross-play | Gap |
|---|---|---:|---:|---:|
| `unident_s` | `traj-full10` | 479.1 | 472.1 | +7.0 |
| | `s2-mix` | 409.2 | 403.3 | +5.8 |
| | `hsp-w1` | 228.1 | 277.0 | -48.9 |
| `random0_medium` | `traj-full10` | 253.1 | 147.9 | +105.2 |
| | `s2-mix` | 240.1 | 230.3 | +9.8 |
| | `hsp-w1` | 44.9 | 35.8 | +9.1 |

Only `traj-full10` on `random0_medium` shows a large convention mismatch. `s2-mix` members
work well together on both layouts. The `hsp-w1` gap does not measure the usual mismatch,
since each member was trained to partner one biased agent rather than itself.

**What each population could reach by choosing among its members, in hindsight:**

| Layout | Population | Best single member | Best member per partner |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | 176.6 | 182.0 |
| | `s2-mix` | 310.2 | 323.0 |
| | `hsp-w1` | 217.7 | 278.7 |
| `random0_medium` | `traj-full10` | 125.5 | 140.5 |
| | `s2-mix` | 148.9 | 156.7 |
| | `hsp-w1` | 115.4 | 154.6 |

The per-partner figure is a reference rather than a strict bound, since switching within
an episode can beat it. On `random0_medium`, `traj-full10` and `s2-mix` with argmax score
above it, though within noise (+2.3 [-6.1, +11.5] and +5.6 [-3.9, +14.5]). On `unident_s`,
TrajeDi members are limited both by this and by being mutually compatible, with almost no self-play/cross-play gap
(479.1 vs. 472.1, against 253.1 vs. 147.9 on `random0_medium`), so there is little to
choose between.

**Argmax beats sampling for ensembles.** Sampling from a mixture of incompatible policies
produces incoherent behavior. Taking the mixture's most probable action is better for every
population with uniform weights on `random0_medium` (e.g. 112.5 vs. 156.4 for `s2-mix`),
while it barely changes the stage-2 baselines (at most 6.4 for any family) and makes MAPPO
worse (-14.3 on `unident_s`, -4.8 on `random0_medium`). Section 4's choice of argmax was
the right one for the ensemble.

**A note on `traj-s10`.** The pool's `traj-S1-s10` folder actually holds 15 members and
`traj-S1-s15` holds 10, in the public pool too, so those labels appear to be swapped.
`traj-s10` is members 1-10 of the apparent 15-member population, used before this was
noticed, and `traj-full10` is the real 10-member one. Both lead to the same conclusions,
and `traj-s10`'s results are kept in `results/overcooked/`.

## Status against the proposal

Done: the adapter is validated, and the method, its baselines and ablations run
end to end on real Overcooked on two layouts, with three populations. TrajeDi, HSP and
MAPPO baselines are ZSC-Eval's own trained agents, so they are at full fidelity.

Not done:

- **Population training.** All policies come from a pre-trained pool; none have been
  trained in this repository for real Overcooked. `scripts/train_population.py` and
  `training/ppo_trainer.py` are gridworld-only. ZSC-Eval's trainers would be the way
  to train new ones.
- **The REINFORCE arm.** `--advantage reinforce` is implemented and tested but has not
  been run at scale on real Overcooked, so actor-critic has not been compared with it.
- **More layouts and seeds.** Only two layouts, three training seeds.
- **COLE and E3T baselines**, which ZSC-Eval's pool has but this evaluation leaves out.
- **Human play** against the ensemble through the Overcooked UI (Section 4), which
  needs an agent wrapper for `ZSC-Eval/zsceval/human_exp/`.
