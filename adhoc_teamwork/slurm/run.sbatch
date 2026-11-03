#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=32gb
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/cmlscratch/%u/adhoc_teamwork/logs/%x-%j.out

# Runs the command given as arguments on a compute node. From the repo root:
#   sbatch -J my-job adhoc_teamwork/slurm/run.sbatch python -m adhoc_teamwork.scripts.foo --bar
#
# Scavenger jobs get preempted and requeued, which restarts this script, so
# anything long has to be resumable (the training and benchmark scripts are).
# Override any setting above on the sbatch command line, e.g. --time=2:00:00.

source "${ADHOC_REPO:-$SLURM_SUBMIT_DIR}/adhoc_teamwork/slurm/env.sh"

echo "=== $(date) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "=== $*"
"$@"
status=$?
echo "=== $(date) finished with status $status"
exit $status
