#!/usr/bin/env bash
# Run/resume the frozen four-arm hierarchical-W2 study from a persistent runtime.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_hierarchical_w2_job.sh PERSIST_ROOT NEW_JOB_DIR OUTPUT_ROOT' >&2
  exit 2
fi

h2_persist=$1
h2_job=$2
h2_output=$3
h2_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
h2_python=${FLUXBIN_PYTHON:-python}

if [ ! -d "$h2_persist" ] || [ ! -d "$h2_persist/models" ]; then
  echo "persistent volume is unavailable: $h2_persist" >&2
  exit 1
fi
if [ -e "$h2_job" ]; then
  echo "job directory already exists: $h2_job" >&2
  exit 1
fi
if [ -n "$(git -C "$h2_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a formal quality run' >&2
  git -C "$h2_root" status --short >&2
  exit 1
fi
if ! command -v "$h2_python" >/dev/null 2>&1; then
  echo "persistent Python is unavailable: $h2_python" >&2
  exit 1
fi

mkdir -p "$h2_output"
mkdir "$h2_job" || exit 1
printf '%s\n' running > "$h2_job/status"
git -C "$h2_root" rev-parse HEAD > "$h2_job/git-revision"
git -C "$h2_root" status --short --branch > "$h2_job/git-status"
command -v "$h2_python" > "$h2_job/python-executable.txt"
"$h2_python" -m pip freeze > "$h2_job/environment-freeze.txt"
nvidia-smi -q > "$h2_job/nvidia-smi.txt" 2>&1

export PYTHONPATH="$h2_root/src:$h2_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

h2_config="$h2_root/configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"
h2_snapshot="$h2_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
h2_calibration="$h2_root/artifacts/qwen3-8b-c4-v1"
h2_protocol="$h2_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-protocol-6109964.json"
h2_tokens="$h2_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
h2_w3="$h2_persist/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1"
h2_config_hash=$(sha256sum "$h2_config" | awk '{print $1}')

if [ -f "$h2_output/config.sha256" ]; then
  if [ "$(awk '{print $1}' "$h2_output/config.sha256")" != "$h2_config_hash" ]; then
    echo 'output root belongs to a different config; refusing mixed resume' >&2
    exit 1
  fi
else
  printf '%s  %s\n' "$h2_config_hash" "$h2_config" > "$h2_output/config.sha256"
fi

"$h2_python" -m pip check > "$h2_job/environment-check.log" 2>&1
h2_code=$?
if [ "$h2_code" -ne 0 ]; then
  printf '%s\n' "$h2_code" > "$h2_job/exit-code"
  printf '%s\n' failed_environment_check > "$h2_job/status"
  exit "$h2_code"
fi

{
  printf '%s  %s\n' \
    96b59535af55f30fe4b3992baa565800bfaaf659975d3e5fb501af0bfc17e482 \
    "$h2_w3/manifest.json"
  printf '%s  %s\n' \
    8b61cbeaba8809b94bc6b7568cb45ede0d941d046c812f57f3156163e42069ae \
    "$h2_protocol"
  printf '%s  %s\n' \
    252938697260d7f7241f26a05b9822c1a2fd5e9ae0168a87e0d5345d4a111ee0 \
    "$h2_tokens"
} | sha256sum --check --strict > "$h2_job/reference-preflight.log" 2>&1
h2_code=$?
if [ "$h2_code" -ne 0 ]; then
  printf '%s\n' "$h2_code" > "$h2_job/exit-code"
  printf '%s\n' failed_reference_preflight > "$h2_job/status"
  exit "$h2_code"
fi

"$h2_python" -u "$h2_root/scripts/run_qwen3_8b_hierarchical_w2_gptq.py" \
  --config "$h2_config" --arm H2.50 --snapshot-root "$h2_snapshot" \
  --calibration-manifest "$h2_calibration/manifest.json" \
  --calibration-tokens "$h2_calibration/tokens.safetensors" \
  --artifact-dir "$h2_output/H2.50/layers" --output "$h2_output/H2.50/result.preflight-unused.json" \
  --validate-only > "$h2_job/input-preflight.log" 2>&1
h2_code=$?
if [ "$h2_code" -ne 0 ]; then
  printf '%s\n' "$h2_code" > "$h2_job/exit-code"
  printf '%s\n' failed_input_preflight > "$h2_job/status"
  exit "$h2_code"
fi

for h2_arm in H2.50 H2.625 H2.75 H2.875; do
  h2_arm_dir="$h2_output/$h2_arm"
  mkdir -p "$h2_arm_dir"
  if [ -f "$h2_arm_dir/result.json" ]; then
    printf 'skip complete arm %s\n' "$h2_arm" >> "$h2_job/progress.log"
    continue
  fi
  h2_command=(
    "$h2_python" -u "$h2_root/scripts/run_qwen3_8b_hierarchical_w2_gptq.py"
    --config "$h2_config"
    --arm "$h2_arm"
    --snapshot-root "$h2_snapshot"
    --calibration-manifest "$h2_calibration/manifest.json"
    --calibration-tokens "$h2_calibration/tokens.safetensors"
    --artifact-dir "$h2_arm_dir/layers"
    --output "$h2_arm_dir/result.json"
  )
  printf '%q ' "${h2_command[@]}" > "$h2_job/command-$h2_arm.txt"
  printf '\n' >> "$h2_job/command-$h2_arm.txt"
  "${h2_command[@]}" > "$h2_job/quant-$h2_arm.log" 2>&1
  h2_code=$?
  if [ "$h2_code" -ne 0 ]; then
    printf '%s\n' "$h2_code" > "$h2_job/exit-code"
    printf 'failed_%s\n' "$h2_arm" > "$h2_job/status"
    exit "$h2_code"
  fi
  printf 'complete arm %s\n' "$h2_arm" >> "$h2_job/progress.log"
done

h2_ppl_command=(
  "$h2_python" -u "$h2_root/scripts/run_qwen3_8b_hierarchical_w2_ppl.py"
  --config "$h2_config"
  --snapshot-root "$h2_snapshot"
  --protocol-manifest "$h2_protocol"
  --token-artifact "$h2_tokens"
  --w3-manifest "$h2_w3/manifest.json"
  --w3-dir "$h2_w3"
  --h2-50-result "$h2_output/H2.50/result.json"
  --h2-50-artifact-dir "$h2_output/H2.50/layers"
  --h2-625-result "$h2_output/H2.625/result.json"
  --h2-625-artifact-dir "$h2_output/H2.625/layers"
  --h2-75-result "$h2_output/H2.75/result.json"
  --h2-75-artifact-dir "$h2_output/H2.75/layers"
  --h2-875-result "$h2_output/H2.875/result.json"
  --h2-875-artifact-dir "$h2_output/H2.875/layers"
  --output "$h2_output/ppl-result.json"
)
printf '%q ' "${h2_ppl_command[@]}" > "$h2_job/command-ppl.txt"
printf '\n' >> "$h2_job/command-ppl.txt"
"${h2_ppl_command[@]}" > "$h2_job/ppl.log" 2>&1
h2_code=$?
printf '%s\n' "$h2_code" > "$h2_job/exit-code"
if [ "$h2_code" -eq 0 ] && [ -f "$h2_output/ppl-result.json" ]; then
  sha256sum "$h2_output"/*/result.json "$h2_output/ppl-result.json" > "$h2_job/results.sha256"
  printf '%s\n' completed_pending_review > "$h2_job/status"
else
  printf '%s\n' failed_ppl > "$h2_job/status"
fi
exit "$h2_code"
