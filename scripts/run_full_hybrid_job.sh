#!/usr/bin/env bash
# Run inside tmux with the project environment's Python on PATH.
set -u
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash scripts/run_full_hybrid_job.sh NEW_JOB_DIR RUNNER_ARGS...' >&2
  exit 2
fi
full_job=$1
shift
full_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$full_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir "$full_job" || exit 1
python -u "$full_root/scripts/run_qwen3_8b_full_hessian_obq_s8.py" "$@" > "$full_job/hybrid.log" 2>&1 &
full_pid=$!
printf '%s\n' "$full_pid" > "$full_job/hybrid.pid"
wait "$full_pid"
full_code=$?
printf '%s\n' "$full_code" > "$full_job/exit-code"
exit "$full_code"
