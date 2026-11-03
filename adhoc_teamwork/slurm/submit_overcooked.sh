#!/bin/bash
# Submits the real-Overcooked experiment: one-off evaluation jobs for each
# layout and population, then a train-then-evaluate chain per training seed.
# Everything is resumable, so re-running this after a failure is safe.
#
# From the repo root:
#   bash adhoc_teamwork/slurm/submit_overcooked.sh
#
# Environment overrides:
#   LAYOUTS, POPULATION, ARMS, SEEDS, N_ITERATIONS, EPISODES_PER_ITERATION,
#   PARTNERS_PER_ITERATION, ONLY (eval|train; default both)
#
# An arm is <estimator>:<train partner set>, e.g. actor_critic:sp.

set -euo pipefail
LAYOUTS="${LAYOUTS:-unident_s random0_medium}"
POPULATION="${POPULATION:-traj-s10}"
ARMS="${ARMS:-actor_critic:sp actor_critic:sp+biased reinforce:sp}"
SEEDS="${SEEDS:-0 1 2}"
N_ITERATIONS="${N_ITERATIONS:-1000}"
EPISODES_PER_ITERATION="${EPISODES_PER_ITERATION:-64}"
PARTNERS_PER_ITERATION="${PARTNERS_PER_ITERATION:-8}"
ONLY="${ONLY:-both}"
ROOT="${ADHOC_SCRATCH:-/cmlscratch/$USER/adhoc_teamwork}/runs"
RUN="adhoc_teamwork/slurm/run.sbatch"
mkdir -p "$ROOT"
[ -f "$ROOT/../local.env" ] && source "$ROOT/../local.env"

for layout in $LAYOUTS; do
    dir="$ROOT/$layout/$POPULATION"
    mkdir -p "$dir"
    if [ "$ONLY" != "train" ]; then
        sbatch --parsable -J "adhoc-baselines-$layout" "$RUN" \
            python -m adhoc_teamwork.scripts.benchmark_overcooked --what baselines --layout "$layout" --out "$ROOT/$layout/baselines"
        sbatch --parsable -J "adhoc-crossplay-$layout" --time=6:00:00 "$RUN" \
            python -m adhoc_teamwork.scripts.build_crossplay_matrix --env overcooked --layout "$layout" --population "$POPULATION" \
            --n-episodes 20 --device cuda --out "$dir/crossplay"
        sbatch --parsable -J "adhoc-population-$layout" "$RUN" \
            python -m adhoc_teamwork.scripts.benchmark_overcooked --what population --layout "$layout" --population "$POPULATION" --out "$dir/population_eval"
    fi
    if [ "$ONLY" != "eval" ]; then
        for arm in $ARMS; do
            estimator="${arm%%:*}"
            partners="${arm##*:}"
            for seed in $SEEDS; do
                out="$dir/$estimator-$partners/seed$seed"
                train_job=$(sbatch --parsable -J "adhoc-train-$layout-$estimator-$partners-$seed" "$RUN" \
                    python -m adhoc_teamwork.scripts.train_adaptive_ensemble --env overcooked --layout "$layout" \
                    --population "$POPULATION" --train-partners "$partners" --advantage "$estimator" \
                    --n-iterations "$N_ITERATIONS" --episodes-per-iteration "$EPISODES_PER_ITERATION" \
                    --partners-per-iteration "$PARTNERS_PER_ITERATION" --embed-dim 32 --encoder-hidden 64 --weighting-hidden 64 \
                    --device cuda --seed "$seed" --out "$out")
                sbatch --parsable -J "adhoc-eval-$layout-$estimator-$partners-$seed" --dependency="afterok:$train_job" "$RUN" \
                    python -m adhoc_teamwork.scripts.benchmark_overcooked --what ensemble --layout "$layout" --population "$POPULATION" \
                    --ensemble "$out/ensemble.pt" --seed "$seed" --out "$out/eval"
            done
        done
    fi
done

# Mirror progress to wandb if configured (see slurm/env.sh). `singleton` keeps
# at most one sync running; it exits on its own once the other jobs are done.
if [ -n "${WANDB_PROJECT:-}" ]; then
    sbatch --parsable -J adhoc-wandb-sync --dependency=singleton --gres=gpu:0 --ntasks=1 --mem=8gb "$RUN" \
        python -m adhoc_teamwork.scripts.sync_wandb --root "$ROOT" --loop 600
fi
