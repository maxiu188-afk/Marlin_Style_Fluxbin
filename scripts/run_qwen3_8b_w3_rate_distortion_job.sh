#!/usr/bin/env bash
# Launch the frozen four-arm PPL comparison inside a durable tmux session.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_w3_rate_distortion_job.sh PERSIST_ROOT NEW_JOB_DIR NEW_OUTPUT_DIR' >&2
  exit 2
fi

ppl_persist=$1
ppl_job=$2
ppl_output=$3
ppl_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ppl_python=${FLUXBIN_PYTHON:-python}

if [ ! -d "$ppl_persist" ] || [ ! -d "$ppl_persist/models" ]; then
  echo "persistent volume is not mounted or has no models directory: $ppl_persist" >&2
  exit 1
fi
if [ -e "$ppl_output" ]; then
  echo "output directory already exists; refusing to overwrite: $ppl_output" >&2
  exit 1
fi
if [ -n "$(git -C "$ppl_root" status --porcelain)" ]; then
  echo "repository worktree is not clean; refusing a formal run" >&2
  git -C "$ppl_root" status --short >&2
  exit 1
fi
if ! command -v "$ppl_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $ppl_python" >&2
  exit 1
fi

# A new directory prevents retries from truncating earlier logs or provenance.
mkdir "$ppl_job" || exit 1
printf '%s\n' running > "$ppl_job/status"
git -C "$ppl_root" rev-parse HEAD > "$ppl_job/git-revision"
git -C "$ppl_root" status --short --branch > "$ppl_job/git-status"
"$ppl_python" -m pip freeze > "$ppl_job/environment-freeze.txt"
command -v "$ppl_python" > "$ppl_job/python-executable.txt"
nvidia-smi -q > "$ppl_job/nvidia-smi.txt" 2>&1

export PYTHONPATH="$ppl_root/src:$ppl_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ppl_config="$ppl_root/configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json"
ppl_snapshot="$ppl_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ppl_protocol="$ppl_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-protocol-6109964.json"
ppl_tokens="$ppl_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
ppl_qbb="$ppl_persist/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1"
ppl_qbb_fp16="$ppl_persist/models/fluxbin/qwen3-8b-hybrid-distilled-step400-fp16-scales-v1"
ppl_gptq="$ppl_persist/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1"

"$ppl_python" -m pip check > "$ppl_job/environment-check.log" 2>&1
ppl_code=$?
if [ "$ppl_code" -ne 0 ]; then
  printf '%s\n' "$ppl_code" > "$ppl_job/exit-code"
  printf '%s\n' failed_environment_check > "$ppl_job/status"
  exit "$ppl_code"
fi

{
  printf '%s  %s\n' \
    96b59535af55f30fe4b3992baa565800bfaaf659975d3e5fb501af0bfc17e482 \
    "$ppl_gptq/manifest.json"
  printf '%s  %s\n' \
    03a3e52e902b6467c61559e0051b6da477e53c298e4aefb47f1715fe80502dff \
    "$ppl_qbb_fp16/result.json"
} | sha256sum --check --strict > "$ppl_job/artifact-preflight.log" 2>&1
ppl_code=$?
if [ "$ppl_code" -ne 0 ]; then
  printf '%s\n' "$ppl_code" > "$ppl_job/exit-code"
  printf '%s\n' failed_artifact_preflight > "$ppl_job/status"
  exit "$ppl_code"
fi

"$ppl_python" -c \
  'import json,sys; from pathlib import Path; from run_qwen3_8b_w3_rate_distortion_ppl import validate_runtime; validate_runtime(json.loads(Path(sys.argv[1]).read_text()))' \
  "$ppl_config" > "$ppl_job/runtime-preflight.log" 2>&1
ppl_code=$?
if [ "$ppl_code" -ne 0 ]; then
  printf '%s\n' "$ppl_code" > "$ppl_job/exit-code"
  printf '%s\n' failed_runtime_preflight > "$ppl_job/status"
  exit "$ppl_code"
fi

ppl_command=(
  "$ppl_python" -u "$ppl_root/scripts/run_qwen3_8b_w3_rate_distortion_ppl.py"
  --config "$ppl_config"
  --snapshot-root "$ppl_snapshot"
  --protocol-manifest "$ppl_protocol"
  --token-artifact "$ppl_tokens"
  --qbb-acceptance "$ppl_qbb/acceptance.json"
  --qbb-result "$ppl_qbb/result.json"
  --qbb-payload-dir "$ppl_qbb/payloads"
  --qbb-fp16-result "$ppl_qbb_fp16/result.json"
  --qbb-fp16-dir "$ppl_qbb_fp16"
  --gptq-manifest "$ppl_gptq/manifest.json"
  --gptq-dir "$ppl_gptq"
  --output-dir "$ppl_output"
)
printf '%q ' "${ppl_command[@]}" > "$ppl_job/command.txt"
printf '\n' >> "$ppl_job/command.txt"

"${ppl_command[@]}" > "$ppl_job/run.log" 2>&1 &
ppl_pid=$!
printf '%s\n' "$ppl_pid" > "$ppl_job/pid"
wait "$ppl_pid"
ppl_code=$?
printf '%s\n' "$ppl_code" > "$ppl_job/exit-code"

if [ "$ppl_code" -eq 0 ] && [ -f "$ppl_output/summary.json" ]; then
  sha256sum "$ppl_output/summary.json" "$ppl_output/summary.md" > "$ppl_job/summary.sha256"
  printf '%s\n' completed_pending_review > "$ppl_job/status"
else
  printf '%s\n' failed > "$ppl_job/status"
fi
exit "$ppl_code"
