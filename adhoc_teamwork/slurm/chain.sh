#!/bin/bash
# Runs commands one after another, stopping at the first that fails. Commands
# are separated by ":::", e.g.
#   bash adhoc_teamwork/slurm/chain.sh python -m a --x 1 ::: python -m b --y 2
# Every step must be resumable, as a requeued job starts again from the first.
set -e
cmd=()
for arg in "$@"; do
    if [ "$arg" == ":::" ]; then
        "${cmd[@]}"
        cmd=()
    else
        cmd+=("$arg")
    fi
done
if [ ${#cmd[@]} -gt 0 ]; then "${cmd[@]}"; fi
