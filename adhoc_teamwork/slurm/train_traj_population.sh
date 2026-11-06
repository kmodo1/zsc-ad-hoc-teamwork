#!/bin/bash
# Train a TrajeDi stage-1 population with ZSC-Eval's own trainer, using the settings
# of ZSC-Eval's shell/train_traj_stage_1.sh. The population file is written to scratch
# so that the policy pool itself stays read-only.
#
#   sbatch -J <name> --ntasks=16 --mem=64gb --time=3-00:00:00 adhoc_teamwork/slurm/run.sbatch \
#       bash adhoc_teamwork/slurm/train_traj_population.sh <layout> [size] [seed]
#
# Takes about a day and cannot resume: a preempted job restarts from the beginning.
# Differences from that script, neither changing what is trained:
#   --dummy_batch_size 5 (not 1): 50 worker processes rather than 250, for 16 CPUs
#   --use_wandb (a store_false flag, so this turns wandb off)
set -euo pipefail
layout=$1
size=${2:-10}
seed=${3:-1001}

yml="$ADHOC_SCRATCH/new_policies/yml/$layout-traj-s$size.yml"
mkdir -p "$(dirname "$yml")"
{
    printf 'traj_adaptive:\n    policy_config_path: %s/policy_config/rnn_policy_config.pkl\n    featurize_type: ppo\n    train: False\n' "$layout"
    for i in $(seq 1 "$size"); do
        printf 'traj%d:\n    policy_config_path: %s/policy_config/mlp_policy_config.pkl\n    featurize_type: ppo\n    train: True\n' "$i" "$layout"
    done
} > "$yml"

ulimit -n 65536 || true
exec python ZSC-Eval/zsceval/scripts/overcooked/train/train_traj.py \
    --env_name Overcooked --algorithm_name traj --experiment_name "traj-new-S1-s${size}" \
    --layout_name "$layout" --num_agents 2 --seed "$seed" --n_training_threads 1 --num_mini_batch 1 \
    --episode_length 400 --num_env_steps 1e7 --reward_shaping_horizon 1e8 \
    --train_env_batch 250 --n_rollout_threads 250 --dummy_batch_size 5 --overcooked_version old \
    --ppo_epoch 15 --entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 6e6 1e7 \
    --save_interval 25 --log_interval 1 --use_eval --eval_interval 20 \
    --n_eval_rollout_threads $((size * 2)) --eval_episodes 10 \
    --stage 1 --traj_entropy_alpha 0.1 --traj_gamma 0.5 \
    --population_yaml_path "$yml" --population_size "$size" --adaptive_agent_name traj_adaptive \
    --use_proper_time_limits --use_wandb
