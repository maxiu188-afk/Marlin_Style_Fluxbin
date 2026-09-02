#!/bin/bash
# Materialize the pinned C4 calibration token artifact. No quantization launch.
#SBATCH --job-name=fluxbin-c4-q32
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/qwen3-32b-c4-calibration/%x-%j.out
#SBATCH --error=logs/qwen3-32b-c4-calibration/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
tokenizer_root="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
config="${project_root}/configs/calibration/qwen3_32b_c4_256x2048_v1.json"
output_dir="${artifact_root}/calibration/qwen3-32b-c4-calibration-256x2048-v1"
tokens="${output_dir}/token_ids.safetensors"
result="${output_dir}/manifest.json"
source_manifest="${output_dir}/source-manifest.txt"

mkdir -p "${output_dir}" "${project_root}/logs/qwen3-32b-c4-calibration"
export HF_HOME="${artifact_root}/huggingface-c4"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${project_root}/src"

for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${tokenizer_root}/tokenizer.json" \
    "${project_root}/scripts/materialize_qwen3_c4_calibration.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_C4_MISSING=${required}" >&2
        exit 1
    fi
done
for path in "${tokens}" "${result}" "${source_manifest}"; do
    if [[ -e "${path}" ]]; then
        echo "FLUXBIN_C4_REFUSES_OVERWRITE=${path}" >&2
        exit 1
    fi
done
if [[ -n "$(git status --short)" ]]; then
    echo "FLUXBIN_C4_DIRTY_CHECKOUT" >&2
    git status --short >&2
    exit 1
fi

echo "FLUXBIN_C4_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

echo "FLUXBIN_C4_STAGE=materialize"
"${model_env}/bin/python" "${project_root}/scripts/materialize_qwen3_c4_calibration.py" \
    --config "${config}" \
    --tokenizer-root "${tokenizer_root}" \
    --tokens "${tokens}" \
    --output "${result}" \
    --source-manifest "${source_manifest}"

"${model_env}/bin/python" - "${result}" "${tokens}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["tokens"]["shape"] == [256, 2048], result
with safe_open(sys.argv[2], framework="pt", device="cpu") as artifact:
    assert artifact.get_tensor("token_ids").shape == (256, 2048), result
print("FLUXBIN_C4_STRUCTURED_RESULT_ACCEPTED")
PY

echo "FLUXBIN_C4_RESULT=${result}"
echo "FLUXBIN_C4_TOKENS=${tokens}"
