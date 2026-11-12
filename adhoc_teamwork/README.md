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
  slurm/        env.sh, run.sbatch, chain.sh and submit_overcooked.sh for the UMIACS
                cluster; analyze_all.sh; train_hsp_pair.sh and train_traj_population.sh
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
140 unit + integration tests (62 of them need the `zsceval` environment and
skip themselves without it, and some of those also need a policy pool) cover:
the real-Overcooked adapter's contract and the ZSC-Eval policy wrapper;
that evaluation partners are disjoint from training partners and sub-policies, for every
training set including those that hold out a generator; each training action rule;
the sign-flip test and Holm correction;
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

# Evaluation jobs, plus one train-then-evaluate job per seed. See the script header
# for options. An arm is <estimator>:<training partners>[:<encoder ablation>[:<action rule>]],
# e.g. ARMS="actor_critic:sp actor_critic:sp:memoryless actor_critic:no-biased:none:argmax".
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
a different checkout pip-installed. `run.sbatch` asks for any Turing or Pascal GPU (RTX
2080 Ti, GTX 1080 Ti, P6000): the jobs are bound by Python environment stepping, and these
are far easier to get than A4000s. Scavenger allows 500 queued jobs per user, which is
why each training run evaluates itself in the same job (`slurm/chain.sh`).

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
evaluation partners (10 distinct agents, at two checkpoints each) unless a table says
otherwise, with 20 episodes in each player position. Returns are sparse team reward, 20
per soup. Files are in `results/overcooked/`.

**Statistics.** Differences are paired over partners, with 95% bootstrap intervals over
the distinct partner agents. Each column of a table asks one question across layouts and
populations, so it is treated as one family of comparisons: p-values come from exact
sign-flip tests over the same agents and are Holm-corrected within the column, and **†**
marks a difference that is significant at 0.05 after that correction. With 10 agents no
p-value can be below 2/1024, so a correction over a whole table could leave nothing
significant by construction. With the 5 E3T agents it is 2/32, so E3T differences are
never significant. The number of distinct partners, not training seeds, is what limits
these tests. Tables without † (cross-play, the gain breakdown, the policies trained here)
are uncorrected. `python -m adhoc_teamwork.scripts.report_tables` prints every corrected
table.

**Partner groups.** Besides the biased agents, every method also played against unseen
agents from other generators: a MEP population, 8 self-play agents, a TrajeDi population,
10 COLE agents (from populations of 25 and 50, not the baseline's 75) and the 5 E3T agents.
A group is left out for a population whose members trained with its agents (ZSC-Eval's
stage-2 agents trained with self-play seeds 1-15 and, through the swapped folder labels,
possibly the TrajeDi population; they never saw `mep-S1-s10`), and a baseline is left out
of a group it trained with or is part of (`experiment.GROUP_SEEN_BY_BASELINES`).

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
amount that depends heavily on the population. With sampled actions the gain survives the
correction in seven of nine cases. With argmax actions, five intervals lie above zero but
none survives: the gains are as large, but vary more between partners.

| Layout | Population | Sampled | Argmax |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | -1.3 [-4.4, +0.7] | +5.5 [-0.4, +13.7] |
| | `s2-mix` | +4.1 [+2.2, +5.9] † | +4.1 [-2.6, +12.1] |
| | `hsp-w1` | +74.8 [+57.6, +94.9] † | +33.0 [+6.5, +52.9] |
| `random0_medium` | `traj-full10` | +49.9 [+28.3, +73.9] † | +27.7 [+5.7, +51.3] |
| | `s2-mix` | +28.1 [+22.4, +34.3] † | +5.9 [+0.8, +11.4] |
| | `hsp-w1` | +45.6 [+21.3, +74.5] † | +37.5 [+11.7, +68.6] |
| `random3` | `traj-full10` | +6.7 [-0.1, +16.7] | +10.6 [-0.4, +28.1] |
| | `s2-mix` | +15.3 [+9.5, +21.1] † | +0.6 [-4.6, +5.8] |
| | `hsp-w1` | +49.4 [+34.3, +64.2] † | +53.0 [+29.3, +76.4] |

**Adaptive against the best baseline, like for like.**

| Layout | Population | Sampled | Argmax |
|---|---|---:|---:|
| `unident_s` | `traj-full10` | -131.0 [-183.5, -78.4] † | -131.9 [-185.3, -78.4] † |
| | `s2-mix` | -4.4 [-5.9, -3.0] † | **+12.4 [+5.1, +20.7] †** |
| | `hsp-w1` | -37.9 [-59.6, -19.8] † | -39.6 [-64.5, -18.1] † |
| `random0_medium` | `traj-full10` | +6.5 [-10.5, +21.3] | +2.0 [-13.8, +15.4] |
| | `s2-mix` | +6.3 [-1.1, +13.2] | **+21.6 [+11.1, +30.8] †** |
| | `hsp-w1` | +8.5 [-4.5, +20.3] | +12.0 [-2.7, +25.4] |
| `random3` | `traj-full10` | -63.5 [-81.9, -42.8] † | -57.3 [-77.0, -36.3] † |
| | `s2-mix` | +6.2 [-2.2, +15.4] | **+16.9 [+9.0, +24.3] †** |
| | `hsp-w1` | -4.3 [-17.8, +11.3] | -5.0 [-20.0, +12.0] |

Only `s2-mix` with argmax beats the best baseline on all three layouts, significantly after
correction, and most of that does not come from adaptation. The same ensemble with
*uniform* weights and argmax already gets +8.3 [-4.9, +20.5], +15.6 [+0.9, +29.2] and
+16.4 [+11.4, +21.8].

On `random3`, `hsp-w1` is statistically level with the best baseline, although none of its
members scores above 39.8 alone: adapting takes it to about the best member per partner in
hindsight (87.3 vs. 88.0).

**Is the gain from observing the partner's trajectory?** The encoder ablations test this.
Each row is the complete encoder minus an ablated one, so a positive number means the
removed part helped:

| Layout | Population | No memory, sampled | No memory, argmax | Partner actions hidden, sampled | Partner actions hidden, argmax |
|---|---|---:|---:|---:|---:|
| `unident_s` | `hsp-w1` | +35.1 [+15.9, +55.3] † | +2.3 [-24.6, +28.6] | +0.8 [-3.9, +6.3] | -2.2 [-12.5, +9.1] |
| | `s2-mix` | +2.8 [+1.4, +4.1] † | +3.3 [-2.5, +10.0] | not run | not run |
| `random0_medium` | `hsp-w1` | +6.9 [+3.8, +10.2] † | +7.8 [+3.4, +12.3] † | -0.1 [-1.8, +1.2] | +0.2 [-1.6, +1.8] |
| | `s2-mix` | +6.9 [+4.1, +10.1] † | +1.5 [-0.6, +4.0] | not run | not run |
| | `traj-full10` | +4.2 [+1.4, +7.2] | -1.6 [-7.6, +3.6] | +0.8 [-1.5, +3.5] | -2.7 [-3.9, -1.6] † |
| `random3` | `hsp-w1` | +6.3 [+2.2, +12.3] † | +14.5 [+3.6, +32.7] † | -2.0 [-3.5, -0.4] | -1.5 [-3.5, +0.4] |
| | `s2-mix` | +5.8 [+3.7, +8.1] † | +6.2 [+2.2, +10.1] | not run | not run |
| | `traj-full10` | +0.7 [-1.6, +2.7] | +3.7 [-0.9, +8.5] | -0.1 [-0.7, +0.6] | +0.6 [-0.6, +2.2] |

- **`hsp-w1` shows real partner adaptation.** Compare
  `results/overcooked/unident_s/hsp-w1/weight_traces_actor_critic-sp+biased.png` with
  the `-memoryless` version. With memory, the weights split by partner within about the
  first 50 steps, and the two checkpoints of one agent tend to go the same way. Without
  memory, all 20 panels look almost the same (std between partners 0.065 vs. 0.010 on
  `unident_s`; the drop is smaller on `random3`, 0.064 vs. 0.045). `s2-mix` and
  `traj-full10` show less of this on every layout (0.009-0.038, against 0.047-0.065 for
  `hsp-w1`), which fits members not specialized to partner styles.
- **Memory helps, but is not most of the gain.** Removing it costs significantly in six
  of eight comparisons with sampled actions, but only two of eight with argmax. Where the
  gain over uniform weights is above 10, the no-memory version keeps 53-92% of it with
  sampled actions and 65% or more with argmax. So much of the benefit is gating between
  sub-policies on the current state, which does include the partner's position and held
  item.
- **The partner's action input adds nothing measurable.** Hiding it makes no detectable
  difference in nine of ten comparisons and slightly helps in the tenth. This does not
  mean information about the partner is useless: weights still split by partner without
  it (std 0.064 on `unident_s`), since memory can infer behavior from the observations.

**Training partners.** The `hsp-w1` results above depend on the training partners
resembling the evaluation partners. Trained against self-play partners only, the gain on
`unident_s` mostly disappears, and on `random3` it shrinks to under 40% and is no longer
clearly above zero, which is the risk Section 9 warns about. It holds up better on
`random0_medium`.

| Layout | Trained against | Adaptive, sampled | Against uniform | Adaptive, argmax | Against uniform |
|---|---|---:|---:|---:|---:|
| `unident_s` | Self-play only | 222.2 | +29.4 [-1.3, +62.7] | 216.2 | -19.9 [-46.3, +6.3] |
| | Self-play + other biased agents | 267.6 | +74.8 [+57.6, +94.9] † | 269.0 | +33.0 [+6.5, +52.9] |
| `random0_medium` | Self-play only | 138.5 | +41.3 [+13.5, +74.7] † | 142.9 | +27.7 [+0.0, +60.3] |
| | Self-play + other biased agents | 142.8 | +45.6 [+21.3, +74.5] † | 152.7 | +37.5 [+11.7, +68.6] |
| `random3` | Self-play only | 56.2 | +18.3 [-3.7, +42.3] | 50.8 | +20.5 [-9.0, +51.7] |
| | Self-play + other biased agents | 87.3 | +49.4 [+34.3, +64.2] † | 83.3 | +53.0 [+29.3, +76.4] † |

**Unseen generators.** The biased agents are one generator of partners. The weighting
networks above trained against self-play agents and, for two populations, other biased
agents from the same generator as the evaluation partners. The next tables ask what
happens with partners from generators they never saw. First, the main runs on the other
partner groups, with argmax actions ("n/a" marks a group left out for that population):

| Against uniform weights | Population | MEP | Self-play | TrajeDi | COLE | E3T |
|---|---|---:|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | +1.6 [-0.8, +4.9] | +13.8 [-6.8, +41.5] | n/a | +3.8 [-1.2, +8.6] | -3.9 [-15.5, +5.7] |
| | `s2-mix` | +5.2 [-1.9, +13.2] | n/a | n/a | +6.3 [+4.6, +8.0] † | +5.0 [+2.9, +7.7] |
| | `hsp-w1` | +36.9 [+18.7, +54.9] † | +12.5 [-6.4, +28.6] | -27.4 [-42.5, -12.2] † | -46.3 [-63.1, -30.0] † | -12.5 [-19.9, -5.1] |
| `random0_medium` | `traj-full10` | +52.0 [+27.9, +80.0] † | +38.5 [+11.1, +72.9] † | n/a | +33.3 [+14.6, +54.7] † | +53.7 [+33.6, +73.0] |
| | `s2-mix` | +8.1 [+2.2, +16.0] | n/a | n/a | +1.3 [-0.9, +3.3] | +6.2 [+3.5, +9.6] |
| | `hsp-w1` | +133.9 [+107.6, +159.3] † | +125.9 [+86.9, +166.4] † | +73.9 [+36.8, +116.2] † | +98.0 [+66.2, +128.1] † | +103.9 [+72.9, +133.5] |
| `random3` | `traj-full10` | +25.6 [+15.9, +36.0] † | +32.4 [+22.7, +42.1] † | n/a | +54.0 [+41.1, +67.4] † | +21.3 [+10.9, +31.5] |
| | `s2-mix` | -11.8 [-26.9, +1.6] | n/a | n/a | +0.3 [-3.5, +6.0] | +4.8 [-5.4, +14.5] |
| | `hsp-w1` | +85.3 [+70.7, +100.7] † | +86.6 [+67.9, +102.7] † | +98.7 [+78.1, +118.7] † | +34.9 [+13.5, +54.4] | -11.5 [-37.2, +14.1] |

| Against the best baseline | Population | MEP | Self-play | TrajeDi | COLE | E3T |
|---|---|---:|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | -3.9 [-13.2, +4.0] | -53.1 [-98.3, -16.3] | n/a | -141.4 [-168.4, -114.8] † | -93.0 [-112.9, -76.0] |
| | `s2-mix` | +5.7 [+1.2, +10.0] | n/a | n/a | -29.0 [-31.2, -26.7] † | -6.9 [-11.6, -2.5] |
| | `hsp-w1` | -12.3 [-20.2, -4.9] | -30.7 [-45.4, -17.5] † | -46.5 [-63.4, -27.0] † | -102.1 [-120.8, -83.4] † | -27.9 [-33.8, -18.8] |
| `random0_medium` | `traj-full10` | +16.3 [+8.6, +24.2] | +21.2 [+9.9, +32.1] | n/a | +0.1 [-3.4, +3.2] | +2.8 [-5.0, +9.6] |
| | `s2-mix` | +15.1 [+6.0, +24.4] | n/a | n/a | +2.9 [-0.5, +6.8] | +6.3 [+1.4, +10.8] |
| | `hsp-w1` | -10.1 [-27.3, +4.3] | -28.9 [-64.4, -1.0] | -19.1 [-50.5, +5.9] | -30.2 [-43.0, -19.0] † | -7.6 [-10.2, -4.9] |
| `random3` | `traj-full10` | -30.4 [-56.4, -3.5] | -2.1 [-22.5, +17.0] | n/a | -50.5 [-66.7, -34.2] † | -108.3 [-127.2, -93.1] |
| | `s2-mix` | +8.1 [-1.7, +16.9] | n/a | n/a | -6.9 [-14.2, -0.8] | -11.9 [-32.7, +4.4] |
| | `hsp-w1` | -25.7 [-36.7, -13.6] | -22.4 [-40.9, -2.9] | -20.8 [-39.2, -1.6] | -45.8 [-60.8, -31.8] † | -79.6 [-94.9, -62.1] |

- **Adapting still helps on most other generators, but not everywhere.** `hsp-w1` gains
  74-134 over uniform weights on every group on `random0_medium`, and 85-99 with MEP,
  self-play and TrajeDi partners on `random3`. On `unident_s`, the same network is
  significantly *worse* than uniform weights with TrajeDi and COLE partners (-27.4 and
  -46.3): what it learned from biased training partners does not carry over there.
- **No population beats the baselines on other generators the way `s2-mix` does on biased
  partners.** On MEP partners `s2-mix` is level with the best baseline after correction
  (+5.7 to +15.1, none significant). On COLE partners it is behind COLE's own baseline on
  `unident_s` (-29.0) and `random3` (-6.9), and level on `random0_medium`; that baseline
  shares the group's generator, much as HSP's shares the biased one. `hsp-w1` trails the
  best baseline on every other group. Its members were each trained for one biased partner.

**Leaving one generator out of training.** `hsp-w1` was retrained against four generators
of training partners (self-play, biased, MEP and TrajeDi agents, none of them evaluation
partners), sampling a generator and then one of its agents. It was also trained four more
times, each leaving one generator out. Each held-out generator is then tested on its own
unseen agents. COLE and E3T appear in no training set. Mean return, argmax actions:

| Layout | Trained against | Biased | MEP | Self-play | TrajeDi | COLE | E3T |
|---|---|---:|---:|---:|---:|---:|---:|
| `unident_s` | *Uniform weights* | 236.0 | 439.6 | 436.7 | 473.6 | 424.9 | 406.2 |
| | Self-play | 216.1 | 497.1 | 470.2 | 485.4 | 388.2 | 396.4 |
| | Self-play + biased | 269.0 | 476.5 | 449.2 | 446.2 | 378.6 | 393.7 |
| | All four | 260.4 | 495.8 | 457.2 | 487.7 | 374.8 | 387.7 |
| | All but self-play | 267.5 | 494.1 | 454.5 | 483.2 | 368.7 | 387.6 |
| | All but biased | 208.4 | 478.8 | 460.9 | 464.7 | 387.4 | 387.1 |
| | All but MEP | 259.2 | 491.1 | 457.3 | 482.0 | 370.9 | 383.7 |
| | All but TrajeDi | 246.4 | 492.9 | 451.7 | 480.8 | 362.2 | 367.2 |
| `random0_medium` | *Uniform weights* | 115.2 | 182.8 | 168.1 | 116.1 | 201.6 | 165.3 |
| | Self-play | 142.9 | 328.4 | 328.5 | 213.8 | 310.8 | 275.7 |
| | Self-play + biased | 152.7 | 316.7 | 294.1 | 190.0 | 299.6 | 269.2 |
| | All four | 147.2 | 326.7 | 326.3 | 225.0 | 309.4 | 275.7 |
| | All but self-play | 148.2 | 321.0 | 304.8 | 212.4 | 306.0 | 275.7 |
| | All but biased | 143.3 | 327.1 | 327.8 | 218.1 | 307.3 | 274.2 |
| | All but MEP | 148.8 | 323.0 | 323.0 | 215.1 | 310.2 | 274.5 |
| | All but TrajeDi | 146.8 | 318.7 | 303.8 | 206.8 | 303.6 | 274.3 |
| `random3` | *Uniform weights* | 30.3 | 27.9 | 25.6 | 0.1 | 100.5 | 78.4 |
| | Self-play | 50.8 | 124.1 | 138.2 | 119.6 | 147.4 | 68.3 |
| | Self-play + biased | 83.3 | 113.2 | 112.2 | 98.8 | 135.3 | 66.9 |
| | All four | 76.4 | 123.2 | 122.8 | 100.9 | 143.1 | 75.4 |
| | All but self-play | 76.0 | 119.5 | 121.3 | 90.6 | 142.5 | 59.6 |
| | All but biased | 66.7 | 121.4 | 124.2 | 103.2 | 150.1 | 51.3 |
| | All but MEP | 75.3 | 122.3 | 125.8 | 92.0 | 143.6 | 69.9 |
| | All but TrajeDi | 80.1 | 123.8 | 123.5 | 103.8 | 132.6 | 74.3 |

The cost of leaving a generator out, on that generator's unseen agents:

| Layout | Held-out generator | Trained with it | Trained without it | With minus without | Without, against uniform |
|---|---|---:|---:|---:|---:|
| `unident_s` | Self-play | 457.2 | 454.5 | +2.7 [-4.2, +11.4] | +17.8 [-2.7, +37.2] |
| | Biased | 260.4 | 208.4 | +52.0 [+23.6, +81.1] | -27.6 [-53.2, -2.9] |
| | MEP | 495.8 | 491.1 | +4.7 [+1.4, +8.9] | +51.5 [+36.1, +67.8] † |
| | TrajeDi | 487.7 | 480.8 | +7.0 [+0.8, +17.7] | +7.2 [+0.4, +18.6] |
| `random0_medium` | Self-play | 326.3 | 304.8 | +21.5 [+8.4, +39.8] | +136.7 [+95.3, +178.8] |
| | Biased | 147.2 | 143.3 | +3.8 [+0.7, +7.3] | +28.1 [+1.1, +60.9] |
| | MEP | 326.7 | 323.0 | +3.7 [+1.2, +6.5] | +140.2 [+116.2, +163.2] † |
| | TrajeDi | 225.0 | 206.8 | +18.3 [+7.8, +30.0] | +90.7 [+45.9, +139.2] † |
| `random3` | Self-play | 122.8 | 121.3 | +1.5 [-7.9, +10.3] | +95.6 [+82.5, +108.3] |
| | Biased | 76.4 | 66.7 | +9.7 [+2.6, +17.1] | +36.4 [+12.8, +60.3] |
| | MEP | 123.2 | 122.3 | +1.0 [-4.0, +4.7] | +94.3 [+78.1, +109.1] † |
| | TrajeDi | 100.9 | 103.8 | -2.9 [-13.9, +8.1] | +103.7 [+80.9, +124.7] † |

- **The self-play-style generators stand in for each other.** A network that never saw MEP,
  TrajeDi or self-play agents still beats uniform weights on them in all nine cases, with
  intervals above zero in eight and five significant after correction (the 8 self-play
  agents cannot reach significance in this table). It loses little against having trained
  with them (-2.9 to +21.5, none significant). Self-play training alone already does about
  as well on them as all four generators.
- **Biased partners are the exception.** Without biased training partners, the network is
  below uniform weights on biased partners on `unident_s` (-27.6 [-53.2, -2.9]), and
  training with them is worth +52.0 there. On the other layouts they are worth less (+3.8
  and +9.7), and the network beats uniform weights even without them.
- **More generators did not buy generalization to new ones.** On the COLE and E3T agents,
  which no training set contains, training on all four generators does about the same as
  self-play alone. Every trained network is below uniform weights on `unident_s` (COLE
  362-388 against 424.9, E3T 367-396 against 406.2) and with E3T partners on `random3`. On
  `random0_medium`, every trained network is far above uniform weights.

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
  `hsp-w1` on `random0_medium` and `random3`, and `s2-mix` on `random3`; only the first two
  survive the correction (see the ablation table).
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

**Training the way it acts.** The main runs train the weighting network by sampling from
the mixture, but Section 4's ensemble takes its most probable action. Three action rules
make training match evaluation (`--action-rule`, the fourth field of an arm):

- `argmax`: Gaussian noise on the weight logits, then the noisy mixture's most probable
  action. The policy gradient is that of the noisy weights, and evaluation drops the noise.
- `anneal` (Saurav's): sample from `softmax(log(mixture) / T)`, with T annealed from 1 to
  0.3. Run on `s2-mix` on two layouts.
- `select` / `select_argmax`: draw one sub-policy with probability equal to its weight and
  act with it. Evaluation (policy selection) then uses the highest-weighted one.

Each cell is the matched rule minus training by sampling, both evaluated the same way:

| Layout | Population | Ensemble, argmax (noisy weights) | Ensemble, argmax (annealed) | Policy selection, sampled | Policy selection, argmax |
|---|---|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | -3.4 [-7.3, -0.4] | - | +0.4 [-0.8, +1.9] | -1.9 [-3.3, -0.7] |
| | `s2-mix` | -3.6 [-8.2, +0.5] | - | -4.4 [-8.5, -1.6] † | -3.8 [-7.4, -0.1] |
| | `hsp-w1` | -4.0 [-15.6, +12.0] | - | +1.2 [-8.8, +9.6] | +0.5 [-22.2, +25.5] |
| `random0_medium` | `traj-full10` | -11.5 [-22.8, +0.2] | - | -6.5 [-10.5, -2.5] | -3.4 [-8.1, +1.2] |
| | `s2-mix` | -1.2 [-3.7, +0.8] | -0.7 [-2.6, +1.1] | -1.5 [-3.7, +0.3] | -9.9 [-18.1, -2.8] |
| | `hsp-w1` | -6.2 [-13.6, -0.5] | - | -5.3 [-7.6, -3.2] † | -8.4 [-12.2, -4.5] † |
| `random3` | `traj-full10` | -4.0 [-8.3, +0.3] | - | -0.5 [-1.8, +1.0] | -5.6 [-8.2, -3.1] † |
| | `s2-mix` | +0.6 [-5.6, +5.9] | +1.4 [-4.4, +6.8] | -7.5 [-13.4, -2.0] | -6.2 [-12.7, -0.4] |
| | `hsp-w1` | -15.0 [-24.4, -7.3] | - | -3.3 [-8.3, +0.4] | -8.0 [-12.6, -3.9] † |

- **Matching the rule never helped.** No matched rule is significantly better anywhere,
  and it is significantly worse in five cells. Training by sampling gives the best weights
  even for argmax actions.
- **The matched rules learn less.** Under the `argmax` and `anneal` rules the weights stay
  closer to uniform than when sampling (for `hsp-w1`, mean weight entropy at the end of
  training is 2.0-2.1 nats against 1.2-1.7, where uniform over 10 members is 2.30), and the
  learned noise scale never moves from its starting 0.5. A small change in the weights only changes the most probable action
  near a tie, so most steps give the network no learning signal, whereas sampling gives one
  on every action.

**Policy selection with argmax.** With the same trained network (each population's main
run), acting with the highest-weighted sub-policy against acting with the weighted mixture:

| Layout | Population | Selection | Ensemble | Ensemble minus selection |
|---|---|---:|---:|---:|
| `unident_s` | `traj-full10` | 175.5 | 176.7 | +1.3 [+0.1, +2.4] |
| | `s2-mix` | 314.1 | 321.0 | +6.9 [-2.4, +16.0] |
| | `hsp-w1` | 250.9 | 269.0 | +18.0 [+9.2, +28.4] † |
| `random0_medium` | `traj-full10` | 144.4 | 142.8 | -1.6 [-6.4, +3.4] |
| | `s2-mix` | 156.7 | 162.3 | +5.6 [+0.1, +11.8] |
| | `hsp-w1` | 143.7 | 152.7 | +9.0 [+1.4, +17.7] |
| `random3` | `traj-full10` | 24.7 | 31.0 | +6.3 [+3.6, +9.7] † |
| | `s2-mix` | 95.9 | 105.3 | +9.4 [+6.8, +12.1] † |
| | `hsp-w1` | 79.0 | 83.3 | +4.3 [+1.7, +7.3] |

Blending beats selecting in eight of nine cases, three significantly, and is never
significantly worse. Training a selector by selecting does not close the gap (above).

**Mixed populations.** Stage-2 generalists (`s2-mix`'s agents), specialists (`hsp-w1`'s)
and TrajeDi members in different ratios, 12 members in all (`experiment.MIXED_POPULATIONS`),
trained against self-play and biased partners. Argmax actions, biased partners:

| Layout | Generalists : specialists : TrajeDi | Uniform | Adaptive (trained sampling) | Adaptive (trained argmax) | Adaptive against uniform | Adaptive against best baseline |
|---|---|---:|---:|---:|---:|---:|
| `unident_s` | 12 : 0 : 0 (`s2-mix`) | 316.9 | 321.0 | 317.4 | +4.1 [-2.6, +12.1] | +12.4 [+5.1, +20.7] |
| | 8 : 4 : 0 | 311.5 | 315.2 | 316.6 | +3.7 [-4.9, +11.8] | +6.6 [-5.7, +17.8] |
| | 4 : 8 : 0 | 246.5 | 316.0 | 310.5 | +69.5 [+38.5, +105.5] † | +7.4 [+0.4, +14.5] |
| | 4 : 4 : 4 | 231.6 | 309.8 | 309.5 | +78.3 [+43.6, +113.9] † | +1.3 [-7.6, +10.4] |
| | 0 : 10 : 0 (`hsp-w1`) | 236.0 | 269.0 | 265.0 | +33.0 [+6.5, +52.9] | -39.6 [-64.5, -18.1] |
| `random0_medium` | 12 : 0 : 0 (`s2-mix`) | 156.4 | 162.3 | 161.2 | +5.9 [+0.8, +11.4] | +21.6 [+11.1, +30.8] |
| | 8 : 4 : 0 | 153.8 | 160.1 | 159.9 | +6.3 [+1.9, +10.7] | +19.3 [+8.2, +29.8] |
| | 4 : 8 : 0 | 149.4 | 152.9 | 154.5 | +3.5 [-4.2, +11.8] | +12.2 [-7.4, +29.6] |
| | 4 : 4 : 4 | 129.2 | 150.9 | 153.5 | +21.7 [+6.7, +37.9] | +10.1 [-2.5, +22.0] |
| | 0 : 10 : 0 (`hsp-w1`) | 115.2 | 152.7 | 146.5 | +37.5 [+11.7, +68.6] | +12.0 [-2.7, +25.4] |
| `random3` | 12 : 0 : 0 (`s2-mix`) | 104.7 | 105.3 | 105.8 | +0.6 [-4.6, +5.8] | +16.9 [+9.0, +24.3] |
| | 8 : 4 : 0 | 98.0 | 104.6 | 102.9 | +6.6 [+1.1, +14.4] | +16.3 [+2.0, +34.2] |
| | 4 : 8 : 0 | 63.9 | 97.3 | 91.9 | +33.4 [+19.9, +50.4] † | +9.0 [-2.5, +22.9] |
| | 4 : 4 : 4 | 40.1 | 88.5 | 89.5 | +48.4 [+29.7, +74.4] † | +0.2 [-10.6, +12.5] |
| | 0 : 10 : 0 (`hsp-w1`) | 30.3 | 83.3 | 68.3 | +53.0 [+29.3, +76.4] | -5.0 [-20.0, +12.0] |

On the generators every mix is held out from (MEP, COLE and E3T together, 25 agents),
the adaptive ensemble against `s2-mix`'s:

| Layout | 8 : 4 : 0 | 4 : 8 : 0 | 4 : 4 : 4 | 0 : 10 : 0 (`hsp-w1`) |
|---|---:|---:|---:|---:|
| `unident_s` | -2.6 [-3.7, -1.5] † | -2.8 [-4.2, -1.4] † | -12.4 [-18.1, -6.8] † | -40.7 [-54.3, -28.2] † |
| `random0_medium` | -4.8 [-6.2, -3.3] † | -11.3 [-14.3, -8.7] † | -11.9 [-15.3, -8.8] † | -26.1 [-32.6, -20.3] † |
| `random3` | -7.7 [-12.3, -3.3] † | -4.5 [-8.9, +0.3] | -3.3 [-9.5, +2.2] | -42.6 [-52.2, -33.3] † |

- **In a mix, the network does adapt.** Specialists drag the uniform ensemble down (316.9
  to 231.6 on `unident_s`), and the learned weights recover nearly all of it (309.8):
  +69.5 and +78.3 over uniform weights on `unident_s`, and +33.4 and +48.4 on `random3`,
  all significant.
- **But no mix beats the all-generalist `s2-mix`.** On biased partners the best mix is
  0.7-5.0 behind it, and on unseen generators every mix is behind it, significantly in
  seven of nine cases. Specialists for biased partners add nothing on top of strong
  generalists, even when the network learns to use them.

**Actor-critic against REINFORCE.** The same main runs trained with the whole-episode
REINFORCE estimator instead:

| Layout | Population | REINFORCE, sampled | Actor-critic minus REINFORCE, sampled | REINFORCE, argmax | Actor-critic minus REINFORCE, argmax |
|---|---|---:|---:|---:|---:|
| `unident_s` | `traj-full10` | 175.7 | -1.1 [-2.3, -0.1] | 174.9 | +1.9 [+0.1, +4.1] |
| | `s2-mix` | 304.8 | -3.6 [-5.4, -1.8] † | 309.7 | +11.3 [+5.7, +17.7] † |
| | `hsp-w1` | 221.1 | +46.6 [+31.7, +64.3] † | 217.0 | +51.9 [+27.8, +73.8] † |
| `random0_medium` | `traj-full10` | 130.0 | +10.8 [+6.2, +15.5] † | 123.9 | +18.8 [+12.8, +25.6] † |
| | `s2-mix` | 131.2 | +9.4 [+6.5, +12.4] † | 136.6 | +25.7 [+20.2, +31.0] † |
| | `hsp-w1` | 126.0 | +16.8 [+13.8, +19.9] † | 125.9 | +26.9 [+18.5, +36.8] † |
| `random3` | `traj-full10` | 27.5 | +0.6 [-2.4, +4.3] | 27.1 | +3.9 [+0.1, +7.9] |
| | `s2-mix` | 97.1 | +0.7 [-7.7, +8.4] | 92.7 | +12.6 [+0.5, +23.7] |
| | `hsp-w1` | 58.8 | +28.5 [+21.5, +36.2] † | 44.5 | +38.8 [+24.7, +54.3] † |

Actor-critic is ahead in 16 of 18 cells, 11 significantly, with the largest gaps where
adapting matters most (`hsp-w1`, +17 to +52). REINFORCE's weights collapse onto one or two
members (weight entropy 0.1-1.0 nats at the end of training, against 1.2-2.3 for
actor-critic), so it adapts less. It is slightly but significantly better in one cell,
`s2-mix` on `unident_s` with sampled actions (-3.6).

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
on real Overcooked on three layouts, with three populations and three mixtures of them.
All seven baselines (MAPPO, TrajeDi, HSP, FCP, MEP, COLE, E3T) are ZSC-Eval's own trained
agents, so they are at full fidelity. Two of the populations were also retrained here with
ZSC-Eval's trainers. Every method is evaluated on partners from six generators, the network
was trained with each generator of training partners held out, policy selection and the
REINFORCE estimator run at scale, and training has been tried with each evaluation's own
action rule. Differences are corrected for multiple comparisons.

Not done:

- **Retraining `s2-mix`.** Each stage-2 agent takes 100M steps, so only the stage-1
  populations were retrained.
- **More layouts, and more distinct partners.** Three layouts, three training seeds per run
  (baselines have five), and one training run per new policy. With 10 distinct partner
  agents per group, the tests cannot resolve small differences after correction; more
  partners would help more than more seeds.
- **A training signal for the argmax rule.** Neither noisy weights nor an annealed
  temperature beat training by sampling. A lower-variance estimator for the argmax rule
  (e.g. one that differentiates through which action wins near ties) is untried.
- **Human play** against the ensemble through the Overcooked UI (Section 4), which
  needs an agent wrapper for `ZSC-Eval/zsceval/human_exp/`.
