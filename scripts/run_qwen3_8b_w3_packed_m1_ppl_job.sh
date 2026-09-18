#!/usr/bin/env bash
# Durable, resumable full WikiText-2 PPL for the final fused packed W3 M=1 path.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_w3_packed_m1_ppl_job.sh PERSIST_ROOT JOB_DIR RESULT_DIR' >&2
  exit 2
fi

ppl_persist=$1
ppl_job=$2
ppl_result=$3
ppl_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ppl_python=${FLUXBIN_PYTHON:?set FLUXBIN_PYTHON to the persistent venv interpreter}
image_reference=${FLUXBIN_IMAGE_REFERENCE:-runpod-default-unresolved}
export FLUXBIN_EXTENSION_CACHE_ROOT=${FLUXBIN_EXTENSION_CACHE_ROOT:-$ppl_persist/cache/torch-extensions/fluxbin-content}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ppl_persist/cache/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ppl_persist/cache/triton}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$(dirname "$ppl_python"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ppl_root/src:$ppl_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ ! -d "$ppl_persist" ] || [ ! -d "$ppl_persist/models" ]; then
  echo "persistent volume is not mounted: $ppl_persist" >&2
  exit 1
fi
if [ -n "$(git -C "$ppl_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a formal PPL run' >&2
  git -C "$ppl_root" status --short >&2
  exit 1
fi
if ! command -v "$ppl_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $ppl_python" >&2
  exit 1
fi

current_revision=$(git -C "$ppl_root" rev-parse HEAD) || exit 1
if [ ! -e "$ppl_job" ]; then
  mkdir -p "$ppl_job" || exit 1
  printf '%s\n' "$current_revision" > "$ppl_job/source-revision"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ppl_job/started-at"
elif [ ! -f "$ppl_job/source-revision" ] || [ "$(cat "$ppl_job/source-revision")" != "$current_revision" ]; then
  echo "job directory belongs to a different or unknown source revision: $ppl_job" >&2
  exit 1
fi
printf '%s\n' running > "$ppl_job/status"

environment_json=$ppl_job/environment/environment.json
if [ ! -f "$environment_json" ]; then
  if [ -e "$ppl_job/environment" ]; then
    echo "incomplete environment directory: $ppl_job/environment" >&2
    exit 1
  fi
  set -o pipefail
  if ! "$ppl_python" "$ppl_root/scripts/record_acceleration_environment.py" \
    --output-dir "$ppl_job/environment" \
    --phase recreated \
    --image-reference "$image_reference" \
    --require-cuda 2>&1 | tee "$ppl_job/environment.log"; then
    set +o pipefail
    printf '%s\n' failed_environment > "$ppl_job/status"
    exit 1
  fi
  set +o pipefail
fi

prewarm_manifest=$ppl_job/w3-prewarm.json
if [ ! -f "$prewarm_manifest" ]; then
  set -o pipefail
  if ! "$ppl_python" "$ppl_root/scripts/prewarm_w3_extensions.py" \
    --output "$prewarm_manifest" 2>&1 | tee "$ppl_job/prewarm.log"; then
    set +o pipefail
    printf '%s\n' failed_prewarm > "$ppl_job/status"
    exit 1
  fi
  set +o pipefail
fi

ppl_config=$ppl_root/configs/evaluation/qwen3_8b_w3_packed_m1_ppl_v1.json
ppl_snapshot=$ppl_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
ppl_protocol=$ppl_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-protocol-6109964.json
ppl_tokens=$ppl_persist/data/qwen3-wikitext2-v1/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors
ppl_layout=$ppl_persist/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1

ppl_command=(
  "$ppl_python" -u "$ppl_root/scripts/run_qwen3_8b_w3_packed_m1_ppl.py"
  --config "$ppl_config"
  --snapshot-root "$ppl_snapshot"
  --protocol-manifest "$ppl_protocol"
  --token-artifact "$ppl_tokens"
  --w3-layout-root "$ppl_layout"
  --w3-layout-manifest-sha256 f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8
  --environment "$environment_json"
  --prewarm-manifest "$prewarm_manifest"
  --output-dir "$ppl_result"
)
if [ -e "$ppl_result" ]; then
  ppl_command+=(--resume)
fi
printf '%q ' "${ppl_command[@]}" > "$ppl_job/command.txt"
printf '\n' >> "$ppl_job/command.txt"

set -o pipefail
"${ppl_command[@]}" 2>&1 | tee -a "$ppl_job/run.log"
ppl_code=${PIPESTATUS[0]}
set +o pipefail
printf '%s\n' "$ppl_code" > "$ppl_job/exit-code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ppl_job/finished-at"
if [ "$ppl_code" -eq 0 ] && [ -f "$ppl_result/summary.json" ]; then
  sha256sum "$ppl_result/summary.json" > "$ppl_job/summary.sha256"
  printf '%s\n' completed_pending_review > "$ppl_job/status"
else
  printf '%s\n' failed > "$ppl_job/status"
fi
exit "$ppl_code"
