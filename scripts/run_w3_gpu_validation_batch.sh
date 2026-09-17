#!/usr/bin/env bash
# Run every pending W3 GPU validation in one rented-instance session.
#
# GPU time is rented by the hour, so the jobs are ordered cheapest-and-most-
# informative first and each one is fail-soft: a job that fails is recorded and
# the batch continues, because a single bad job must not waste the rental. The
# batch is resumable -- a job whose result JSON already exists is skipped -- so
# a disconnect costs only the job that was running.
set -u

if [ "$#" -ne 2 ]; then
  echo 'Usage: bash scripts/run_w3_gpu_validation_batch.sh BATCH_JOB_DIR BATCH_RESULT_DIR' >&2
  exit 2
fi

batch_job=$1
batch_result=$2
batch_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
batch_python=${FLUXBIN_PYTHON:?set FLUXBIN_PYTHON to the venv interpreter, not system python}
snapshot_root=${FLUXBIN_SNAPSHOT_ROOT:?set FLUXBIN_SNAPSHOT_ROOT}
gptq_root=${FLUXBIN_GPTQ_ROOT:?set FLUXBIN_GPTQ_ROOT}
layout_root=${FLUXBIN_W3_LAYOUT_ROOT:?set FLUXBIN_W3_LAYOUT_ROOT}
layout_sha=${FLUXBIN_W3_LAYOUT_MANIFEST_SHA256:?set FLUXBIN_W3_LAYOUT_MANIFEST_SHA256}

# The 2026-09-17 retry1 failure was a quoting bug that selected system Python
# before the model even loaded, so resolve and prove the interpreter up front.
if ! command -v "$batch_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $batch_python" >&2
  exit 1
fi
if ! "$batch_python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "interpreter has no working CUDA torch: $batch_python" >&2
  "$batch_python" -c 'import sys,torch; print(sys.executable, torch.__version__)' >&2 || true
  exit 1
fi
if [ -n "$(git -C "$batch_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a formal batch' >&2
  git -C "$batch_root" status --short >&2
  exit 1
fi

mkdir -p "$batch_job" "$batch_result" || exit 1
revision=$(git -C "$batch_root" rev-parse HEAD)
printf '%s\n' "$revision" > "$batch_job/source-revision"
date -u +%Y-%m-%dT%H:%M:%SZ > "$batch_job/started-at"

environment_json=$batch_result/environment/environment.json
if [ ! -f "$environment_json" ]; then
  if [ -e "$batch_result/environment" ]; then
    echo "environment directory exists without environment.json: $batch_result/environment" >&2
    exit 1
  fi
  set -o pipefail
  if ! "$batch_python" "$batch_root/scripts/record_acceleration_environment.py" \
    --output-dir "$batch_result/environment" \
    --phase recreated \
    --image-reference "${FLUXBIN_IMAGE_REFERENCE:-runpod-default-unresolved}" \
    --require-cuda 2>&1 | tee "$batch_job/environment.log"; then
    set +o pipefail
    echo 'environment capture failed; nothing else can run' >&2
    exit 1
  fi
  set +o pipefail
fi

# Nsight has been blocked by ERR_NVGPUCTRPERM on every host so far. Recheck it
# once: if this instance allows counters, profiling the kernel is worth more
# than anything else in the batch and should be added before the rental ends.
{
  echo "== ncu availability =="
  command -v ncu || echo 'ncu not on PATH'
  ncu --version 2>&1 | head -3 || true
  echo "== minimal counter smoke =="
  ncu --metrics sm__cycles_elapsed.avg --target-processes application-only \
    "$batch_python" -c 'import torch; torch.zeros(8, device="cuda").sum().cpu()' 2>&1 | tail -20
} > "$batch_job/nsight-recheck.log" 2>&1 || true

run_job() {
  local name=$1; shift
  local output=$1; shift
  if [ -f "$output" ]; then
    printf '%s\n' "skipped (result exists)" > "$batch_job/$name.status"
    echo "[$name] skipped; $output already exists"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  echo "[$name] starting"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$batch_job/$name.started-at"
  set -o pipefail
  "$@" 2>&1 | tee "$batch_job/$name.log"
  local code=${PIPESTATUS[0]}
  set +o pipefail
  printf '%s\n' "$code" > "$batch_job/$name.exit-code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$batch_job/$name.finished-at"
  if [ "$code" -eq 0 ]; then
    printf '%s\n' completed > "$batch_job/$name.status"
  else
    # Fail-soft on purpose: keep the rental working on the remaining jobs.
    printf '%s\n' "failed ($code)" > "$batch_job/$name.status"
    echo "[$name] FAILED with $code; continuing with the rest of the batch" >&2
  fi
  return 0
}

# 1. Bank-conflict probe: minutes, and it decides whether the kernel rewrite
#    should target shared-memory conflicts at all.
run_job bank-conflict-probe "$batch_result/bank-conflict-probe/result.json" \
  "$batch_python" "$batch_root/scripts/run_w3_lut_bank_conflict_probe.py" \
  --environment "$environment_json" \
  --w3-layout-root "$layout_root" \
  --w3-layout-manifest-sha256 "$layout_sha" \
  --output "$batch_result/bank-conflict-probe/result.json"

# 2. 46-cell split-G / row-tile sweep. Already implemented and already run on
#    RTX PRO 4500; never run on A100, where the occupancy trade-off differs.
run_job split-sweep "$batch_result/split-sweep/result.json" \
  "$batch_python" "$batch_root/scripts/run_w3_lut_bf16_split_trial.py" \
  --environment "$environment_json" \
  --gptq-root "$gptq_root" \
  --w3-layout-root "$layout_root" \
  --w3-layout-manifest-sha256 "$layout_sha" \
  --output "$batch_result/split-sweep/result.json"

# 3. Structural full model with fusion OFF: the same-session control for job 4.
#    Without it the fusion delta would be a cross-session comparison.
run_job full-model-stock "$batch_result/full-model-stock/result.json" \
  "$batch_python" "$batch_root/scripts/run_qwen3_8b_w3_full_m1_trial.py" \
  --protocol "$batch_root/configs/acceleration/qwen3_8b_w3_full_m1_v2.json" \
  --snapshot-root "$snapshot_root" \
  --w3-layout-root "$layout_root" \
  --w3-layout-manifest-sha256 "$layout_sha" \
  --environment "$environment_json" \
  --output "$batch_result/full-model-stock/result.json"

# 4. Same protocol with the RMSNorm/RoPE fusions enabled.
run_job full-model-fused "$batch_result/full-model-fused/result.json" \
  "$batch_python" "$batch_root/scripts/run_qwen3_8b_w3_full_m1_trial.py" \
  --protocol "$batch_root/configs/acceleration/qwen3_8b_w3_fused_full_m1_v2.json" \
  --snapshot-root "$snapshot_root" \
  --w3-layout-root "$layout_root" \
  --w3-layout-manifest-sha256 "$layout_sha" \
  --environment "$environment_json" \
  --output "$batch_result/full-model-fused/result.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "$batch_job/finished-at"
"$batch_python" "$batch_root/scripts/summarize_w3_gpu_validation_batch.py" \
  --batch-result "$batch_result" --job-dir "$batch_job" \
  --output "$batch_result/batch-summary.json" 2>&1 | tee "$batch_job/summary.log"
