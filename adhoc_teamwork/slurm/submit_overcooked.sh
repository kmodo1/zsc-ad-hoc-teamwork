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
#   PARTNERS_PER_ITERATION, TEMPERATURE_END (final temperature of the anneal rule, default 0.3), DEPENDENCY (passed as --dependency to every job but
#   evaluations of new training runs, e.g. afterok:123, and the jobs cancel
#   themselves if it fails; sbatch has no environment variable for this),
#   TRAIN_DEPENDENCY (the same, for training jobs only; overrides DEPENDENCY),
#   and ONLY: which jobs to submit, any of
#     baselines   baselines, sampling and with argmax actions (once per layout)
#     crossplay   the population's cross-play matrix
#     population  each sub-policy and the uniform ensembles
#     train       a training run per arm and seed, then its evaluation
#     reeval      re-evaluate the finished training runs of ARMS and SEEDS (after
#                 adding partner groups or methods; only new cells are computed)
#   default: "baselines crossplay population train".
#
# An arm is <estimator>:<train partner set>[:<encoder ablation>[:<action rule>]],
# e.g. actor_critic:sp, actor_critic:sp:memoryless or actor_critic:no-biased:none:argmax.
# The anneal rule (e.g. actor_critic:sp+biased:none:anneal) samples actions at a temperature
# annealed from 1 to TEMPERATURE_END, to train the weights for the argmax rule.
# The run directory is <estimator>-<partners>[-<ablation>][-<rule>], leaving out
# the defaults (no ablation, sampling).

set -euo pipefail
LAYOUTS="${LAYOUTS:-unident_s random0_medium}"
POPULATION="${POPULATION:-traj-s10}"
ARMS="${ARMS:-actor_critic:sp actor_critic:sp+biased reinforce:sp}"
SEEDS="${SEEDS:-0 1 2}"
N_ITERATIONS="${N_ITERATIONS:-1000}"
EPISODES_PER_ITERATION="${EPISODES_PER_ITERATION:-64}"
PARTNERS_PER_ITERATION="${PARTNERS_PER_ITERATION:-8}"
TEMPERATURE_END="${TEMPERATURE_END:-0.3}"
ONLY="${ONLY:-baselines crossplay population train}"
ROOT="${ADHOC_SCRATCH:-/cmlscratch/$USER/adhoc_teamwork}/runs"
RUN="adhoc_teamwork/slurm/run.sbatch"
mkdir -p "$ROOT"
[ -f "$ROOT/../local.env" ] && source "$ROOT/../local.env"

wants() { [[ " $ONLY " == *" $1 "* ]]; }
dependency_args() { [ -n "$1" ] && echo "--dependency=$1 --kill-on-invalid-dep=yes" || true; }
DEP=$(dependency_args "${DEPENDENCY:-}")
TRAIN_DEP=$(dependency_args "${TRAIN_DEPENDENCY:-${DEPENDENCY:-}}")

for layout in $LAYOUTS; do
    dir="$ROOT/$layout/$POPULATION"
    mkdir -p "$dir"
    if wants baselines; then
        sbatch --parsable -J "adhoc-baselines-$layout" $DEP "$RUN" \
            python -m adhoc_teamwork.scripts.benchmark_overcooked --what baselines --layout "$layout" --out "$ROOT/$layout/baselines"
        sbatch --parsable -J "adhoc-baselines-argmax-$layout" $DEP "$RUN" \
            python -m adhoc_teamwork.scripts.benchmark_overcooked --what baselines --argmax-actions --layout "$layout" --out "$ROOT/$layout/baselines_argmax"
    fi
    if wants crossplay; then
        sbatch --parsable -J "adhoc-crossplay-$layout-$POPULATION" --time=6:00:00 $DEP "$RUN" \
            python -m adhoc_teamwork.scripts.build_crossplay_matrix --env overcooked --layout "$layout" --population "$POPULATION" \
            --n-episodes 20 --device cuda --out "$dir/crossplay"
    fi
    if wants population; then
        sbatch --parsable -J "adhoc-population-$layout-$POPULATION" $DEP "$RUN" \
            python -m adhoc_teamwork.scripts.benchmark_overcooked --what population --layout "$layout" --population "$POPULATION" --out "$dir/population_eval"
    fi
    if wants train || wants reeval; then
        for arm in $ARMS; do
            IFS=: read -r estimator partners ablation rule <<< "$arm"
            ablation="${ablation:-none}"
            rule="${rule:-sample}"
            arm_name="$estimator-$partners"
            [ "$ablation" != "none" ] && arm_name="$arm_name-$ablation"
            [ "$rule" != "sample" ] && arm_name="$arm_name-$rule"
            for seed in $SEEDS; do
                out="$dir/$arm_name/seed$seed"
                job_name="$layout-$POPULATION-$arm_name-$seed"
                eval_cmd=(python -m adhoc_teamwork.scripts.benchmark_overcooked --what ensemble --layout "$layout" --population "$POPULATION"
                    --ensemble "$out/ensemble.pt" --seed "$seed" --out "$out/eval")
                if wants reeval; then
                    if [ -f "$out/ensemble.pt" ]; then
                        sbatch --parsable -J "adhoc-eval-$job_name" $DEP "$RUN" "${eval_cmd[@]}"
                    else
                        echo "not trained yet, skipping: $out" >&2
                    fi
                    continue
                fi
                # One job trains, then evaluates: both resume after preemption, and
                # it halves the number of queued jobs (scavenger allows 500 per user).
                sbatch --parsable -J "adhoc-train-$job_name" $TRAIN_DEP "$RUN" bash adhoc_teamwork/slurm/chain.sh \
                    python -m adhoc_teamwork.scripts.train_adaptive_ensemble --env overcooked --layout "$layout" \
                    --population "$POPULATION" --train-partners "$partners" --advantage "$estimator" --encoder-ablation "$ablation" \
                    --action-rule "$rule" --temperature-end "$TEMPERATURE_END" --n-iterations "$N_ITERATIONS" --episodes-per-iteration "$EPISODES_PER_ITERATION" \
                    --partners-per-iteration "$PARTNERS_PER_ITERATION" --embed-dim 32 --encoder-hidden 64 --weighting-hidden 64 \
                    --device cuda --seed "$seed" --out "$out" ::: "${eval_cmd[@]}"
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
