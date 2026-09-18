#!/usr/bin/env bash
# Run only H2.875, then score W3/H2.50/H2.875 on the frozen PPL protocol.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_hierarchical_w2_endpoint_job.sh PERSIST_ROOT NEW_JOB_DIR OUTPUT_ROOT' >&2
  exit 2
fi

ep_persist=$1
ep_job=$2
ep_output=$3
ep_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ep_python=${FLUXBIN_PYTHON:-python}
ep_policy=${FLUXBIN_EXECUTION_POLICY:-formal-a100}

case "$ep_policy" in
  formal-a100|same-device-quality) ;;
  *) echo "unsupported FLUXBIN_EXECUTION_POLICY: $ep_policy" >&2; exit 2 ;;
esac
if [ -e "$ep_job" ]; then
  echo "job directory already exists: $ep_job" >&2
  exit 1
fi
if [ -n "$(git -C "$ep_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a quality run' >&2
  git -C "$ep_root" status --short >&2
  exit 1
fi
if ! command -v "$ep_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $ep_python" >&2
  exit 1
fi

mkdir "$ep_job" || exit 1
printf '%s\n' running > "$ep_job/status"
printf '%s\n' "$ep_policy" > "$ep_job/execution-policy"
git -C "$ep_root" rev-parse HEAD > "$ep_job/git-revision"
"$ep_python" -m pip freeze > "$ep_job/environment-freeze.txt"
nvidia-smi -q > "$ep_job/nvidia-smi.txt" 2>&1

export PYTHONPATH="$ep_root/src:$ep_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ep_config="$ep_root/configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"
ep_snapshot="$ep_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ep_calibration="$ep_root/artifacts/qwen3-8b-c4-v1"
ep_protocol="$ep_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-protocol-6109964.json"
ep_tokens="$ep_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
ep_w3="$ep_persist/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1"

if [ ! -f "$ep_output/H2.50/result.json" ]; then
  printf '%s\n' missing_completed_H2.50 > "$ep_job/status"
  exit 1
fi
if [ -e "$ep_output/ppl-endpoints-result.json" ]; then
  printf '%s\n' endpoint_result_already_exists > "$ep_job/status"
  exit 1
fi

if [ ! -f "$ep_output/H2.875/result.json" ]; then
  ep_quant=(
    "$ep_python" -u "$ep_root/scripts/run_qwen3_8b_hierarchical_w2_gptq.py"
    --config "$ep_config"
    --arm H2.875
    --execution-policy "$ep_policy"
    --snapshot-root "$ep_snapshot"
    --calibration-manifest "$ep_calibration/manifest.json"
    --calibration-tokens "$ep_calibration/tokens.safetensors"
    --artifact-dir "$ep_output/H2.875/layers"
    --output "$ep_output/H2.875/result.json"
  )
  printf '%q ' "${ep_quant[@]}" > "$ep_job/command-H2.875.txt"
  printf '\n' >> "$ep_job/command-H2.875.txt"
  mkdir -p "$ep_output/H2.875"
  "${ep_quant[@]}" > "$ep_job/quant-H2.875.log" 2>&1
  ep_code=$?
  if [ "$ep_code" -ne 0 ]; then
    printf '%s\n' "$ep_code" > "$ep_job/exit-code"
    printf '%s\n' failed_H2.875 > "$ep_job/status"
    exit "$ep_code"
  fi
fi

ep_ppl=(
  "$ep_python" -u "$ep_root/scripts/run_qwen3_8b_hierarchical_w2_ppl.py"
  --config "$ep_config"
  --evaluation-scope endpoint-only
  --execution-policy "$ep_policy"
  --snapshot-root "$ep_snapshot"
  --protocol-manifest "$ep_protocol"
  --token-artifact "$ep_tokens"
  --w3-manifest "$ep_w3/manifest.json"
  --w3-dir "$ep_w3"
  --h2-50-result "$ep_output/H2.50/result.json"
  --h2-50-artifact-dir "$ep_output/H2.50/layers"
  --h2-875-result "$ep_output/H2.875/result.json"
  --h2-875-artifact-dir "$ep_output/H2.875/layers"
  --output "$ep_output/ppl-endpoints-result.json"
)
printf '%q ' "${ep_ppl[@]}" > "$ep_job/command-ppl.txt"
printf '\n' >> "$ep_job/command-ppl.txt"
"${ep_ppl[@]}" > "$ep_job/ppl.log" 2>&1
ep_code=$?
printf '%s\n' "$ep_code" > "$ep_job/exit-code"
if [ "$ep_code" -eq 0 ] && [ -f "$ep_output/ppl-endpoints-result.json" ]; then
  sha256sum \
    "$ep_output/H2.50/result.json" \
    "$ep_output/H2.875/result.json" \
    "$ep_output/ppl-endpoints-result.json" > "$ep_job/results.sha256"
  if [ "$ep_policy" = formal-a100 ]; then
    printf '%s\n' completed_endpoint_pending_review > "$ep_job/status"
  else
    printf '%s\n' completed_cross_device_endpoint_pending_review > "$ep_job/status"
  fi
else
  printf '%s\n' failed_endpoint_ppl > "$ep_job/status"
fi
exit "$ep_code"
