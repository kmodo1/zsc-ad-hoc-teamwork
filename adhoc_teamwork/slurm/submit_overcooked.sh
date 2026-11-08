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
#   PARTNERS_PER_ITERATION, ONLY (eval|train; default both), and TRAIN_DEPENDENCY
#   (passed as --dependency to the training jobs, e.g. afterok:123; sbatch has no
#   environment variable for this)
#
# An arm is <estimator>:<train partner set>[:<encoder ablation>], e.g.
# actor_critic:sp or actor_critic:sp:memoryless.

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
            IFS=: read -r estimator partners ablation <<< "$arm"
            ablation="${ablation:-none}"
            arm_name="$estimator-$partners"
            [ "$ablation" != "none" ] && arm_name="$arm_name-$ablation"
            for seed in $SEEDS; do
                out="$dir/$arm_name/seed$seed"
                train_job=$(sbatch --parsable -J "adhoc-train-$layout-$POPULATION-$arm_name-$seed" ${TRAIN_DEPENDENCY:+--dependency="$TRAIN_DEPENDENCY"} "$RUN" \
                    python -m adhoc_teamwork.scripts.train_adaptive_ensemble --env overcooked --layout "$layout" \
                    --population "$POPULATION" --train-partners "$partners" --advantage "$estimator" --encoder-ablation "$ablation" \
                    --n-iterations "$N_ITERATIONS" --episodes-per-iteration "$EPISODES_PER_ITERATION" \
                    --partners-per-iteration "$PARTNERS_PER_ITERATION" --embed-dim 32 --encoder-hidden 64 --weighting-hidden 64 \
                    --device cuda --seed "$seed" --out "$out")
                sbatch --parsable -J "adhoc-eval-$layout-$POPULATION-$arm_name-$seed" --dependency="afterok:$train_job" "$RUN" \
                    python -m adhoc_teamwork.scripts.benchmark_overcooked --what ensemble --layout "$layout" --population "$POPULATION" \
                    --ensemble "$out/ensemble.pt" --seed "$seed" --out "$out/eval"
            done
        done
    fi
done

# Mirror progress to wandb if configured (see slurm/env.sh), unless a sync is
# already queued; it finds new runs by itself and exits once other jobs are done.
if [ -n "${WANDB_PROJECT:-}" ] && [ -z "$(squeue -h -u "$USER" -n adhoc-wandb-sync)" ]; then
    sbatch --parsable -J adhoc-wandb-sync --dependency=singleton --gres=gpu:0 --ntasks=1 --mem=8gb "$RUN" \
        python -m adhoc_teamwork.scripts.sync_wandb --root "$ROOT" --loop 600
fi
