#!/bin/bash
# Environment for running this package on the UMIACS cluster. Sourced by
# run.sbatch. Submit jobs from the repository root.
#
# Override any of these in your shell before calling sbatch.

export ADHOC_REPO="${ADHOC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$ADHOC_REPO/adhoc_teamwork" ]; then
    echo "ADHOC_REPO=$ADHOC_REPO is not the repository root (submit from there, or set ADHOC_REPO)" >&2
    exit 1
fi

# Checkpoints and logs go to scratch: the home quota is too small for them.
export ADHOC_SCRATCH="${ADHOC_SCRATCH:-/cmlscratch/$USER/adhoc_teamwork}"
# Personal settings that do not belong in the repository (e.g. WANDB_ENTITY and
# WANDB_PROJECT, which turn on scripts/sync_wandb.py) go in this file.
[ -f "$ADHOC_SCRATCH/local.env" ] && source "$ADHOC_SCRATCH/local.env"
export WANDB_DIR="${WANDB_DIR:-$ADHOC_SCRATCH/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ADHOC_SCRATCH/wandb/.cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$ADHOC_SCRATCH/wandb/.data}"
mkdir -p "$WANDB_DIR"

export ZSC_POLICY_POOL="${ZSC_POLICY_POOL:-$HOME/ZSC-Eval/zsceval/policy_pool}"

# A venv layered on the `zsceval` conda env, adding only pytest. Create with:
#   /path/to/envs/zsceval/bin/python -m venv --system-site-packages $ADHOC_SCRATCH/venv
#   $ADHOC_SCRATCH/venv/bin/pip install pytest
export PATH="$ADHOC_SCRATCH/venv/bin:$PATH"

# PYTHONPATH comes first on sys.path, so this selects the vendored ZSC-Eval
# even if a different checkout is pip-installed into the conda env.
export PYTHONPATH="$ADHOC_REPO:$ADHOC_REPO/ZSC-Eval${PYTHONPATH:+:$PYTHONPATH}"

# Env stepping is single-threaded Python; keep inference from oversubscribing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$ADHOC_REPO" || exit 1
