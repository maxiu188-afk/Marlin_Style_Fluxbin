#!/usr/bin/env bash
# Offline-rotate Qwen3-8B, quantize the single H2.50 arm, then score W3 and the
# rotated arm on the frozen PPL protocol. The unrotated 27.932835 baseline is
# carried in from the accepted endpoint run and is not re-scored here.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_hierarchical_w2_rotated_job.sh PERSIST_ROOT NEW_JOB_DIR OUTPUT_ROOT' >&2
  exit 2
fi

rt_persist=$1
rt_job=$2
rt_output=$3
rt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rt_python=${FLUXBIN_PYTHON:-python}
rt_policy=${FLUXBIN_EXECUTION_POLICY:-formal-a100}

case "$rt_policy" in
  formal-a100|same-device-quality) ;;
  *) echo "unsupported FLUXBIN_EXECUTION_POLICY: $rt_policy" >&2; exit 2 ;;
esac
if [ ! -d "$rt_persist" ] || [ ! -d "$rt_persist/models" ]; then
  echo "persistent volume is unavailable: $rt_persist" >&2
  exit 1
fi
if [ -e "$rt_job" ]; then
  echo "job directory already exists: $rt_job" >&2
  exit 1
fi
if [ -n "$(git -C "$rt_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a quality run' >&2
  git -C "$rt_root" status --short >&2
  exit 1
fi
if ! command -v "$rt_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $rt_python" >&2
  exit 1
fi

mkdir -p "$rt_output"
mkdir "$rt_job" || exit 1
printf '%s\n' running > "$rt_job/status"
printf '%s\n' "$rt_policy" > "$rt_job/execution-policy"
git -C "$rt_root" rev-parse HEAD > "$rt_job/git-revision"
git -C "$rt_root" status --short --branch > "$rt_job/git-status"
command -v "$rt_python" > "$rt_job/python-executable.txt"
"$rt_python" -m pip freeze > "$rt_job/environment-freeze.txt"
nvidia-smi -q > "$rt_job/nvidia-smi.txt" 2>&1

export PYTHONPATH="$rt_root/src:$rt_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

rt_config="$rt_root/configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
rt_snapshot="$rt_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
rt_calibration="$rt_root/artifacts/qwen3-8b-c4-v1"
rt_protocol="$rt_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-protocol-6109964.json"
rt_tokens="$rt_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
rt_w3="$rt_persist/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1"
rt_config_hash=$(sha256sum "$rt_config" | awk '{print $1}')

if [ -f "$rt_output/config.sha256" ]; then
  if [ "$(awk '{print $1}' "$rt_output/config.sha256")" != "$rt_config_hash" ]; then
    echo 'output root belongs to a different config; refusing mixed resume' >&2
    exit 1
  fi
else
  printf '%s  %s\n' "$rt_config_hash" "$rt_config" > "$rt_output/config.sha256"
fi
if [ -f "$rt_output/execution-policy" ]; then
  if [ "$(sed -n '1p' "$rt_output/execution-policy")" != "$rt_policy" ]; then
    echo 'output root belongs to a different execution policy; refusing mixed resume' >&2
    exit 1
  fi
else
  printf '%s\n' "$rt_policy" > "$rt_output/execution-policy"
fi

rt_fail() {
  printf '%s\n' "$2" > "$rt_job/exit-code"
  printf '%s\n' "$1" > "$rt_job/status"
  exit "$2"
}

"$rt_python" -m pip check > "$rt_job/environment-check.log" 2>&1 \
  || rt_fail failed_environment_check $?

{
  printf '%s  %s\n' \
    96b59535af55f30fe4b3992baa565800bfaaf659975d3e5fb501af0bfc17e482 \
    "$rt_w3/manifest.json"
  printf '%s  %s\n' \
    8b61cbeaba8809b94bc6b7568cb45ede0d941d046c812f57f3156163e42069ae \
    "$rt_protocol"
  printf '%s  %s\n' \
    252938697260d7f7241f26a05b9822c1a2fd5e9ae0168a87e0d5345d4a111ee0 \
    "$rt_tokens"
} | sha256sum --check --strict > "$rt_job/reference-preflight.log" 2>&1 \
  || rt_fail failed_reference_preflight $?

if [ -e "$rt_output/ppl-rotated-result.json" ]; then
  printf '%s\n' rotated_result_already_exists > "$rt_job/status"
  exit 0
fi

if [ ! -f "$rt_output/H2.50-rotated/result.json" ]; then
  mkdir -p "$rt_output/H2.50-rotated"
  rt_quant=(
    "$rt_python" -u "$rt_root/scripts/run_qwen3_8b_hierarchical_w2_rotated_gptq.py"
    --config "$rt_config"
    --execution-policy "$rt_policy"
    --snapshot-root "$rt_snapshot"
    --calibration-manifest "$rt_calibration/manifest.json"
    --calibration-tokens "$rt_calibration/tokens.safetensors"
    --artifact-dir "$rt_output/H2.50-rotated/layers"
    --output "$rt_output/H2.50-rotated/result.json"
  )
  printf '%q ' "${rt_quant[@]}" > "$rt_job/command-quant.txt"
  printf '\n' >> "$rt_job/command-quant.txt"
  "${rt_quant[@]}" > "$rt_job/quant-rotated.log" 2>&1 \
    || rt_fail failed_rotated_quantization $?
  printf 'complete rotated H2.50\n' >> "$rt_job/progress.log"
else
  printf 'skip complete rotated H2.50\n' >> "$rt_job/progress.log"
fi

rt_ppl=(
  "$rt_python" -u "$rt_root/scripts/run_qwen3_8b_hierarchical_w2_rotated_ppl.py"
  --config "$rt_config"
  --execution-policy "$rt_policy"
  --snapshot-root "$rt_snapshot"
  --protocol-manifest "$rt_protocol"
  --token-artifact "$rt_tokens"
  --w3-manifest "$rt_w3/manifest.json"
  --w3-dir "$rt_w3"
  --rotated-result "$rt_output/H2.50-rotated/result.json"
  --rotated-artifact-dir "$rt_output/H2.50-rotated/layers"
  --output "$rt_output/ppl-rotated-result.json"
)
printf '%q ' "${rt_ppl[@]}" > "$rt_job/command-ppl.txt"
printf '\n' >> "$rt_job/command-ppl.txt"
"${rt_ppl[@]}" > "$rt_job/ppl.log" 2>&1
rt_code=$?
printf '%s\n' "$rt_code" > "$rt_job/exit-code"
if [ "$rt_code" -eq 0 ] && [ -f "$rt_output/ppl-rotated-result.json" ]; then
  sha256sum \
    "$rt_output/H2.50-rotated/result.json" \
    "$rt_output/ppl-rotated-result.json" > "$rt_job/results.sha256"
  if [ "$rt_policy" = formal-a100 ]; then
    printf '%s\n' completed_rotated_pending_review > "$rt_job/status"
  else
    printf '%s\n' completed_cross_device_rotated_pending_review > "$rt_job/status"
  fi
else
  printf '%s\n' failed_rotated_ppl > "$rt_job/status"
fi
exit "$rt_code"
