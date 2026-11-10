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
  slurm/        env.sh, run.sbatch and submit_overcooked.sh for the UMIACS cluster;
                analyze_all.sh; train_hsp_pair.sh and train_traj_population.sh
                (ZSC-Eval's own trainers, for policies trained here)
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
110 unit + integration tests (40 of them need the `zsceval` environment and
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

Or fetch one layout's checkpoints into an existing pool root, verifying each file's
SHA-256:

```bash
python -m adhoc_teamwork.scripts.download_policy_pool --layout random3 --out $ZSC_POLICY_POOL
```

This skips the pool's `policy_config/*.pkl` files on purpose, because unpickling a
downloaded file can run arbitrary code. Generate them with ZSC-Eval's
`shell/store_config.sh`, or copy them from a copy you already trust. Checkpoints
themselves load with `weights_only=True`.

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

# Tables and plots. Safe to run while jobs are still going. analyze_all.sh does
# every layout and population, and copies the summaries into results/overcooked/.
python -m adhoc_teamwork.scripts.analyze_overcooked --root $ADHOC_SCRATCH/runs --layout random0_medium
sbatch -J adhoc-analyze --gres=gpu:0 adhoc_teamwork/slurm/run.sbatch bash adhoc_teamwork/slurm/analyze_all.sh
```

**Training new policies.** ZSC-Eval's own trainers, wrapped for Slurm. They write their
runs to `$ZSC_RESULTS_DIR` (set by `slurm/env.sh`, on scratch) and cannot resume, so a
preempted run restarts from the beginning:

```bash
# One HSP pair (biased agent + its teammate) with the reward weights of the pool's
# hsp<exp>, and a new random seed. About an hour with 16 CPUs.
sbatch -J pair --ntasks=16 adhoc_teamwork/slurm/run.sbatch bash adhoc_teamwork/slurm/train_hsp_pair.sh random3 20
# A 10-member TrajeDi stage-1 population. About a day.
sbatch -J traj --ntasks=16 --mem=64gb --time=3-00:00:00 adhoc_teamwork/slurm/run.sbatch \
    bash adhoc_teamwork/slurm/train_traj_population.sh random3 10
# Collect finished runs for the hsp-w1-new and traj-new10 populations.
python -m adhoc_teamwork.scripts.extract_new_policies --layout random3
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

## Running real Overcooked locally

The cluster is not needed for evaluation or the tests. On a Linux or WSL machine with
[uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv venv --python 3.9 --seed venv
grep -v '^-e' ZSC-Eval/requirements.txt > /tmp/requirements.txt
uv pip install --python venv/bin/python "setuptools<66" "wheel<0.40"
uv pip install --python venv/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" -r /tmp/requirements.txt pytest pyyaml matplotlib
export PYTHONPATH=$PWD:$PWD/ZSC-Eval OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

This uses CUDA 12.8 rather than the 11.8 in `environment.yml`, for newer GPUs. Replaying the
validation this way matches the cluster's run closely (r = 1.000 over 240 pairings on each
layout, mean absolute difference 0.15).

Use `--device cpu` for evaluation, and run several jobs at once instead. One cell of 20
episodes took 2.8 s on CPU against 4.8 s on an RTX 5080 for a single policy, and 6.6 s against
12.5 s for a 12-member ensemble: the networks are small and each step moves a small batch, so
transfers outweigh any speedup, and the environment itself is Python on the CPU. Training was
not timed. Add `--partner-groups biased` to `benchmark_overcooked` for just the primary
comparison.

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
`random0_medium` (r = 0.995 over 240 pairings), and 63.8 vs. 64.4 on `random3` (r = 0.986
over 320 pairings, including COLE and E3T). COLE and E3T also reproduce on the other two
layouts: 267.4 vs. 268.5 on `unident_s` (r = 0.992) and 131.9 vs. 130.6 on `random0_medium`
(r = 0.997), over 80 pairings each. A second replay on different hardware gives the same
result (see **Running real Overcooked locally**).

**Layouts.** `unident_s` and `random0_medium`, plus `random3`, where self-play collapses (20.6)
while stage-2 agents reach 75-92, and picking the best of the 12 `s2-mix` members for each
partner beats the best family by about 30% (118.7 vs. 91.6). `small_corridor` was the other candidate, but returns there top out
around 2.4 soups per episode.

**Baselines.** All ZSC-Eval's own trained agents, 5 seeds each, sampled / argmax:

| Layout | MAPPO | TrajeDi | HSP | FCP | MEP | COLE | E3T |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unident_s` | 179.8 / 165.5 | 298.9 / 301.7 | 290.5 / 293.7 | 304.7 / 307.3 | **305.6 / 308.6** | 253.2 / 240.7 | 275.4 / 268.6 |
| `random0_medium` | 118.4 / 113.5 | 103.8 / 104.0 | **134.3 / 140.8** | 118.3 / 115.6 | 116.3 / 119.1 | 125.8 / 126.1 | 125.6 / 130.9 |
| `random3` | 20.6 / 18.1 | 77.6 / 73.6 | 87.2 / 84.7 | 88.6 / **88.3** | **91.6** / 85.9 | 75.1 / 71.3 | 76.5 / 73.7 |

COLE (its largest, 75-member population) and E3T are not the best baseline on any layout.

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
| `random3` | `traj-full10` | 21.4 | 28.1 | 20.4 | 31.0 |
| | `s2-mix` | 82.5 | 97.8 | 104.7 | **105.2** |
| | `hsp-w1` | 38.0 | 87.3 | 30.3 | 83.3 |
| | *Best baseline (MEP, FCP)* | | *91.6* | | *88.3* |

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
| `random3` | `traj-full10` | +6.7 [-0.1, +16.6] | +10.6 [-0.4, +28.1] |
| | `s2-mix` | +15.3 [+9.5, +21.1] | +0.5 [-4.6, +5.8] |
| | `hsp-w1` | +49.4 [+34.3, +64.2] | +53.0 [+29.3, +76.4] |

**Adaptive against the best baseline, like for like.**

| Layout | Population | Sampled | Argmax |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | -131.0 [-183.5, -78.4] | -131.9 [-185.3, -78.4] |
| | `s2-mix` | -4.4 [-5.9, -3.0] | **+12.4 [+5.1, +20.7]** |
| | `hsp-w1` | -37.9 [-59.6, -19.8] | -39.6 [-64.5, -18.1] |
| `random0_medium` | `traj-full10` | +6.5 [-10.5, +21.3] | +2.0 [-13.8, +15.4] |
| | `s2-mix` | +6.3 [-1.1, +13.2] | **+21.6 [+11.1, +30.8]** |
| | `hsp-w1` | +8.5 [-4.5, +20.3] | +12.0 [-2.7, +25.4] |
| `random3` | `traj-full10` | -63.5 [-81.9, -42.8] | -57.3 [-77.0, -36.3] |
| | `s2-mix` | +6.2 [-2.2, +15.4] | **+16.9 [+9.0, +24.3]** |
| | `hsp-w1` | -4.3 [-17.8, +11.3] | -5.0 [-20.0, +12.0] |

Only `s2-mix` with argmax beats the best baseline on all three layouts, and most of that
does not come from adaptation. The same ensemble with *uniform* weights and argmax already
gets +8.3 [-4.9, +20.5], +15.6 [+0.9, +29.2] and +16.4 [+11.4, +21.8].

On `random3`, `hsp-w1` is statistically level with the best baseline, although none of its
members scores above 39.8 alone: adapting takes it to about the best member per partner in
hindsight (87.3 vs. 88.0).

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
| `random3` | `hsp-w1` | +6.3 [+2.2, +12.3] | +14.5 [+3.6, +32.7] | -2.0 [-3.6, -0.5] | -1.5 [-3.5, +0.4] |
| | `s2-mix` | +5.8 [+3.7, +8.1] | +6.2 [+2.2, +10.1] | not run | not run |
| | `traj-full10` | +0.7 [-1.6, +2.7] | +3.7 [-0.9, +8.5] | -0.1 [-0.7, +0.6] | +0.6 [-0.6, +2.2] |

- **`hsp-w1` shows real partner adaptation.** Compare
  `results/overcooked/unident_s/hsp-w1/weight_traces_actor_critic-sp+biased.png` with
  the `-memoryless` version. With memory, the weights split by partner within about the
  first 50 steps, and the two checkpoints of one agent tend to go the same way. Without
  memory, all 20 panels look almost the same (std between partners 0.065 vs. 0.010 on
  `unident_s`; the drop is smaller on `random3`, 0.064 vs. 0.045). `s2-mix` and
  `traj-full10` show less of this on every layout (0.009-0.038, against 0.047-0.065 for
  `hsp-w1`), which fits members not specialized to partner styles.
- **Memory helps, but is not most of the gain.** Where the gain over uniform weights is
  above 10, the no-memory version keeps 53-92% of it with sampled actions and 65% or more
  with argmax. So much of the benefit is gating between sub-policies on the current
  state, which does include the partner's position and held item.
- **The partner's action input adds nothing measurable.** Hiding it makes no detectable
  difference in eight of ten comparisons and slightly helps in the other two. This does
  not mean information about the partner is useless: weights still split by partner
  without it (std 0.064 on `unident_s`), since memory can infer behavior from the
  observations.

**Training partners.** The `hsp-w1` results above depend on the training partners
resembling the evaluation partners. Trained against self-play partners only, the gain on
`unident_s` mostly disappears, and on `random3` it shrinks to under 40% and is no longer
clearly above zero, which is the risk Section 9 warns about. It holds up better on
`random0_medium`.

| Layout | Trained against | Adaptive, sampled | Against uniform | Adaptive, argmax | Against uniform |
|---|---|---:|---:|---:|---:|
| `unident_s` | Self-play only | 222.2 | +29.4 [-1.3, +62.7] | 216.2 | -19.9 [-46.3, +6.3] |
| | Self-play + other biased agents | 267.6 | +74.8 [+57.6, +94.9] | 269.0 | +33.0 [+6.5, +52.9] |
| `random0_medium` | Self-play only | 138.5 | +41.3 [+13.5, +74.7] | 142.9 | +27.7 [+0.0, +60.3] |
| | Self-play + other biased agents | 142.8 | +45.6 [+21.3, +74.5] | 152.7 | +37.5 [+11.7, +68.6] |
| `random3` | Self-play only | 56.2 | +18.3 [-3.7, +42.3] | 50.8 | +20.5 [-9.0, +51.7] |
| | Self-play + other biased agents | 87.3 | +49.4 [+34.3, +64.2] | 83.3 | +53.0 [+29.3, +76.4] |

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
| `random3` | `traj-full10` | 194.5 | 114.7 | +79.8 |

`traj-full10` shows a large convention mismatch on `random0_medium` and `random3`, but
not on `unident_s`. Cross-play for `s2-mix` and `hsp-w1` has not been run on `random3`; on the other two
layouts, `s2-mix` members work well together. The `hsp-w1` gap does not measure the usual mismatch,
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
| `random3` | `traj-full10` | 28.5 | 32.8 |
| | `s2-mix` | 104.6 | 118.7 |
| | `hsp-w1` | 39.8 | 88.0 |

The per-partner figure is a reference rather than a strict bound, since switching within
an episode can beat it. On `random0_medium`, `traj-full10` and `s2-mix` with argmax score
above it, though within noise (+2.3 [-6.1, +11.5] and +5.6 [-3.9, +14.5]). On `unident_s`,
TrajeDi members are limited both by this and by being mutually compatible, with almost no self-play/cross-play gap
(479.1 vs. 472.1, against 253.1 vs. 147.9 on `random0_medium`), so there is little to
choose between.

**Where the gain comes from.** Each cell is a paired difference on the biased partners.
This first table needs no trained network, and adds single members with argmax actions.
It replaces an earlier version evaluated on a local machine, which differed from it by
at most 3.2 in any cell.

| Layout | Population | Ensembling, sampled | Argmax on the ensemble | Argmax on one member | Difference |
|---|---|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | +2.1 [+0.6, +3.8] | -4.7 [-12.4, +1.1] | -1.4 [-3.2, +0.0] | -3.3 [-9.7, +1.7] |
| | `s2-mix` | -4.7 [-7.1, -1.3] | +19.8 [+7.3, +30.4] | +0.1 [-2.5, +2.5] | +19.7 [+8.0, +30.0] |
| | `hsp-w1` | +21.5 [+11.8, +30.6] | +43.2 [+19.7, +64.7] | -6.6 [-10.1, -3.1] | +49.8 [+28.1, +69.6] |
| `random0_medium` | `traj-full10` | +11.6 [+4.7, +18.5] | +24.1 [+12.4, +36.1] | -5.5 [-7.2, -3.9] | +29.7 [+17.8, +41.6] |
| | `s2-mix` | -3.1 [-9.3, +2.0] | +43.8 [+31.0, +56.0] | +1.0 [-1.5, +3.4] | +42.8 [+31.1, +54.0] |
| | `hsp-w1` | +22.9 [+14.7, +29.8] | +18.1 [+9.6, +26.5] | -4.0 [-5.8, -2.2] | +22.1 [+12.6, +31.5] |
| `random3` | `traj-full10` | +5.4 [+2.6, +8.6] | -1.0 [-6.8, +3.2] | -3.3 [-5.5, -1.7] | +2.3 [-1.9, +6.2] |
| | `s2-mix` | -3.6 [-7.0, +0.3] | +22.2 [+15.2, +28.9] | -4.6 [-7.8, -1.7] | +26.8 [+21.4, +32.2] |
| | `hsp-w1` | +11.4 [+2.9, +20.0] | -7.6 [-14.7, -0.4] | -5.9 [-8.6, -3.1] | -1.8 [-8.7, +6.3] |

The weighting steps with argmax actions. `python -m adhoc_teamwork.scripts.gain_breakdown`
prints both tables, the same steps for sampled actions, and the vote ensemble.

| Layout | Population | State-based weights | Memory | Both |
|---|---|---:|---:|---:|
| `unident_s` | `traj-full10` | not run | not run | +5.5 [-0.4, +13.7] |
| | `s2-mix` | +0.9 [-1.2, +2.8] | +3.3 [-2.5, +10.0] | +4.1 [-2.6, +12.1] |
| | `hsp-w1` | +30.7 [+1.4, +62.6] | +2.3 [-24.6, +28.6] | +33.0 [+6.5, +52.9] |
| `random0_medium` | `traj-full10` | +29.3 [+10.8, +50.3] | -1.6 [-7.6, +3.6] | +27.7 [+5.7, +51.3] |
| | `s2-mix` | +4.4 [-0.4, +9.8] | +1.6 [-0.6, +4.0] | +6.0 [+0.8, +11.4] |
| | `hsp-w1` | +29.7 [+6.0, +57.9] | +7.8 [+3.4, +12.3] | +37.5 [+11.7, +68.6] |
| `random3` | `traj-full10` | +6.9 [-0.3, +20.3] | +3.7 [-0.9, +8.5] | +10.6 [-0.4, +28.1] |
| | `s2-mix` | -5.7 [-8.9, -2.2] | +6.2 [+2.2, +10.1] | +0.5 [-4.6, +5.8] |
| | `hsp-w1` | +38.4 [+15.7, +61.3] | +14.5 [+3.6, +32.7] | +53.0 [+29.3, +76.4] |

- **Argmax helps ensembles, not single policies.** Argmax changes a single member by -6.6 to
  +1.0, but helps the uniform ensemble by clearly more than that in six of nine cases. The
  exceptions are `traj-full10` on `unident_s` and `random3`, and `hsp-w1` on `random3`. So
  the gain is not from acting deterministically as such. One possible reason, not tested
  here, is that the most probable action of a mixture acts like a vote between members,
  while sampling follows each of them in turn.
- **`s2-mix` gains from argmax ensembling, and little from weights.** Its members are
  already strong, and sampling from them gains nothing.
- **`hsp-w1`, and `traj-full10` on `random0_medium`, gain at every step but memory.** Weights
  add about 28-38 on top of argmax, mostly without memory, and `hsp-w1` on `random3` gains
  most of its 53.0 the same way. Memory's gain with argmax is only clearly above zero for
  `hsp-w1` on `random0_medium` and `random3`, and `s2-mix` on `random3`.
- **Nothing helps `traj-full10` on `unident_s`,** which has almost nothing to choose between.

These intervals have no correction for multiple comparisons, and the steps depend on their
order: weights gain more with sampled actions, where argmax has not already taken its share.

**Policies trained here.** To check that these results are not specific to one set of
pretrained agents, `hsp-w1` and `traj-full10` were retrained with ZSC-Eval's own trainers
and new random seeds (see **Training new policies**). `hsp-w1-new` uses the same 10 biased
reward styles as `hsp-w1`, and its ensembles train against the same partners (the pool's
biased agents, not the new ones). Each policy was trained once, for 10M steps.

| Layout | Population | Mean member | Best member | Best member per partner | Uniform | Adaptive, sampled | Adaptive, argmax | Adaptive vs. uniform, sampled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `unident_s` | `hsp-w1` | 171.3 | 217.7 | 278.7 | 192.8 | 267.6 | 269.0 | +74.8 [+57.6, +94.9] |
| | `hsp-w1-new` | 180.0 | 236.6 | 286.0 | 201.8 | 253.9 | 271.7 | +52.1 [+40.8, +63.6] |
| | `traj-full10` | 173.8 | 176.6 | 182.0 | 175.9 | 174.5 | 176.7 | -1.3 [-4.4, +0.7] |
| | `traj-new10` | 175.3 | 178.3 | 181.9 | 173.9 | 174.9 | 174.1 | +1.0 [-1.9, +3.7] |
| `random0_medium` | `hsp-w1` | 74.2 | 115.4 | 154.6 | 97.2 | 142.8 | 152.7 | +45.6 [+21.3, +74.5] |
| | `hsp-w1-new` | 85.3 | 141.1 | 164.7 | 99.2 | 143.2 | 153.7 | +44.0 [+18.5, +72.8] |
| | `traj-full10` | 79.3 | 125.5 | 140.5 | 90.9 | 140.8 | 142.8 | +49.9 [+28.3, +73.9] |
| | `traj-new10` | 110.8 | 136.2 | 154.2 | 127.4 | 143.8 | 144.5 | +16.4 [+6.1, +27.7] |
| `random3` | `hsp-w1` | 26.5 | 39.8 | 88.0 | 38.0 | 87.3 | 83.3 | +49.4 [+34.3, +64.2] |
| | `hsp-w1-new` | 17.9 | 40.4 | 80.5 | 23.2 | 84.4 | 85.5 | +61.2 [+38.1, +85.2] |
| | `traj-full10` | 16.0 | 28.5 | 32.8 | 21.4 | 28.1 | 31.0 | +6.7 [-0.1, +16.6] |
| | `traj-new10` | 19.0 | 25.8 | 37.7 | 27.8 | 42.6 | 37.1 | +14.8 [+6.3, +23.4] |

- **The main results replicate.** The new specialists again gain a lot from adapting, are
  again statistically level with the best baseline on `random3` (-7.2 [-22.7, +10.6]
  sampled, -2.8 [-17.1, +13.1] argmax) and below it on `unident_s`, and memory again helps
  with sampled actions (+17.2 [+13.4, +20.0], +6.8 [+4.0, +9.7] and +9.2 [+6.4, +12.1]).
  The new TrajeDi population again gains nothing on `unident_s` and stays far below the
  baselines on `random3`.
- **One difference, on `unident_s`.** The new specialists are a little stronger on their
  own, so uniform weights with argmax already reach 265.6, and adapting adds only
  +6.1 [-4.8, +18.6] with argmax. Their weights also depend less on the partner there
  (std 0.021 vs. 0.065).
- **The new TrajeDi members are stronger on `random0_medium`** (110.8 vs. 79.3 on average),
  so adapting has less to add over uniform weights, and ends up at about the same return.

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

Done: the adapter is validated, and the method, its baselines and ablations run end to end
on real Overcooked on three layouts, with three populations. All seven baselines (MAPPO,
TrajeDi, HSP, FCP, MEP, COLE, E3T) are ZSC-Eval's own trained agents, so they are at full
fidelity. Two of the populations were also retrained here with ZSC-Eval's trainers.

Not done:

- **Training with the action rule Section 4 describes.** The weighting network trains by
  sampling from the mixture (`training/train_adaptive_ensemble.py`), but Section 4's method
  takes the argmax, so argmax results use weights tuned for sampled behavior. Its effect
  has not been measured.
- **Policy selection with argmax.** It has only run with sampled actions, so it should
  not be compared with the argmax ensembles.
- **Retraining `s2-mix`.** Each stage-2 agent takes 100M steps, so only the stage-1
  populations were retrained.
- **The REINFORCE arm.** `--advantage reinforce` is implemented and tested but has not
  been run at scale on real Overcooked, so actor-critic has not been compared with it.
- **More layouts and seeds.** Three layouts, three training seeds, and one training run per
  new policy.
- **Human play** against the ensemble through the Overcooked UI (Section 4), which
  needs an agent wrapper for `ZSC-Eval/zsceval/human_exp/`.
