#!/bin/bash
# Train one HSP pair (biased agent w0 + teammate w1) with ZSC-Eval's own trainer,
# using the settings of ZSC-Eval's shell/train_bias_agents.sh for the "old" layouts.
#
#   sbatch -J <name> --ntasks=16 adhoc_teamwork/slurm/run.sbatch \
#       bash adhoc_teamwork/slurm/train_hsp_pair.sh <layout> <exp> [replica]
#
# <exp> picks the biased reward weights, the same way ZSC-Eval's seeds do, so hsp<exp>
# has the same reward as the pool's hsp<exp>. [replica] (default 1) only changes the
# random seed, so this is a new policy trained on that same reward.
#
# Careful: several ZSC-Eval flags are store_false, so passing them turns things OFF.
#   --use_recurrent_policy: feed-forward, like the pool's HSP agents
#   --share_policy:         separate networks for w0 and w1
#   --use_wandb:            no wandb, since ZSC-Eval would log to its own project
set -euo pipefail
layout=$1
exp=$2
replica=${3:-1}
seed=$((exp + 1000 * replica))
offset=$((-1000 * replica))

w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
case "$layout" in
    random0_medium) w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]" ;;
    random3 | unident_s | random1) w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]" ;;
    *) echo "train_hsp_pair.sh: no weights set for layout $layout" >&2; exit 1 ;;
esac

ulimit -n 65536 || true
exec python ZSC-Eval/zsceval/scripts/overcooked/train/train_bias_agent.py \
    --env_name Overcooked --algorithm_name mappo --experiment_name "hsp-new-r${replica}-${exp}" \
    --layout_name "$layout" --num_agents 2 --seed "$seed" --w0_offset="$offset" \
    --n_training_threads 1 --n_rollout_threads 100 --dummy_batch_size 2 --num_mini_batch 1 \
    --episode_length 400 --num_env_steps 1e7 --reward_shaping_horizon 1e7 --overcooked_version old \
    --ppo_epoch 15 --entropy_coefs 0.2 0.05 0.001 --entropy_coef_horizons 0 6e6 1e7 \
    --use_hsp --w0 "$w0" --w1 "$w1" --share_policy --random_index \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 \
    --use_wandb
