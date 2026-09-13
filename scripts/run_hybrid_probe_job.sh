#!/usr/bin/env bash
# Run inside tmux after activating the container-disk venv.
set -u
if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_hybrid_probe_job.sh SNAPSHOT CALIBRATION_DIR NEW_JOB_DIR' >&2
  exit 2
fi
probe_snapshot=$1
probe_calibration=$2
probe_job=$3
probe_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$probe_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Require a new directory so retries cannot truncate a prior log or result.
mkdir "$probe_job" || exit 1
python -u "$probe_root/scripts/run_qwen3_8b_hybrid_compensation_probe.py" \
  --snapshot-root "$probe_snapshot" --calibration-dir "$probe_calibration" \
  --output-dir "$probe_job/artifacts" --execute > "$probe_job/probe.log" 2>&1 &
probe_pid=$!
printf '%s\n' "$probe_pid" > "$probe_job/probe.pid"
wait "$probe_pid"
probe_code=$?
printf '%s\n' "$probe_code" > "$probe_job/exit-code"
exit "$probe_code"
