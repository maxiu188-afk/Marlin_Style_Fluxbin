#!/bin/bash
# Calibrated single-Linear gate. No automatic full-model, PPL, or backend job.
#SBATCH --job-name=fluxbin-obq-q32-s8
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen3-32b-single-linear-hessian-obq-s8/%x-%j.out
#SBATCH --error=logs/qwen3-32b-single-linear-hessian-obq-s8/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
snapshot="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
calibration_dir="${artifact_root}/calibration/qwen3-32b-c4-calibration-256x2048-v1"
calibration_manifest="${calibration_dir}/manifest.json"
calibration_tokens="${calibration_dir}/token_ids.safetensors"
config="${project_root}/configs/experiments/qwen3_32b_single_linear_hessian_obq_s8_v2.json"
results_dir="${artifact_root}/results/qwen3-32b-single-linear-hessian-obq-s8-v2"
stem="fluxbin-obq-qwen3-32b-single-linear-s8-v2-${SLURM_JOB_ID}"
result="${results_dir}/${stem}.json"
payload="${results_dir}/${stem}.safetensors"
source_manifest="${results_dir}/${stem}-source-manifest.txt"

mkdir -p "${results_dir}" "${project_root}/logs/qwen3-32b-single-linear-hessian-obq-s8"
task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/${stem}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"
export HF_HOME="${qbb_artifact_root}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${project_root}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${snapshot}/config.json" \
    "${snapshot}/model.safetensors.index.json" \
    "${calibration_manifest}" \
    "${calibration_tokens}" \
    "${project_root}/src/fluxbin_style/hessian_obq.py" \
    "${project_root}/scripts/run_qwen3_single_linear_hessian_obq_s8.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_OBQ_MISSING=${required}" >&2
        exit 1
    fi
done
for path in "${result}" "${payload}" "${source_manifest}"; do
    if [[ -e "${path}" ]]; then
        echo "FLUXBIN_OBQ_REFUSES_OVERWRITE=${path}" >&2
        exit 1
    fi
done
if [[ -n "$(git status --short)" ]]; then
    echo "FLUXBIN_OBQ_DIRTY_CHECKOUT" >&2
    git status --short >&2
    exit 1
fi

echo "FLUXBIN_OBQ_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

echo "FLUXBIN_OBQ_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=clean\n'
    printf 'model_revision=%s\n' "${revision}"
    printf 'calibration_manifest_sha256=%s\n' "$(sha256sum "${calibration_manifest}" | cut -d' ' -f1)"
    printf 'calibration_tokens_sha256=%s\n' "$(sha256sum "${calibration_tokens}" | cut -d' ' -f1)"
    sha256sum \
        "${config}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${project_root}/src/fluxbin_style/hessian_obq.py" \
        "${project_root}/src/fluxbin_style/two_base_rank1.py" \
        "${project_root}/scripts/run_qwen3_single_linear_hessian_obq_s8.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_single_linear_hessian_obq_s8.sbatch"
} > "${source_manifest}"

echo "FLUXBIN_OBQ_STAGE=execute"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_single_linear_hessian_obq_s8.py" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --calibration-manifest "${calibration_manifest}" \
    --calibration-tokens "${calibration_tokens}" \
    --payload "${payload}" \
    --output "${result}" \
    --source-manifest "${source_manifest}"

"${model_env}/bin/python" - "${result}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "completed_pending_review", result
assert result["calibration"]["calibration_sequences"] == 256, result
assert result["calibration"]["sequence_length"] == 2048, result
assert result["calibration"]["activation_rows"] == 256 * 2048, result
assert result["contract"]["arms_share_global_payload"] is False, result
assert result["decision"]["pure_arm_preserved_independently"], result
assert result["reconstruction"]["hybrid_maximum_delta_outside_selected_columns"] == 0, result
for arm in ("pure_two_base_obq", "hessian_salient_hybrid_s8_obq"):
    assert all(math.isfinite(value) for value in result["reconstruction"][arm].values()), result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN_OBQ_SINGLE_LINEAR_READY_FOR_REVIEW")
PY

echo "FLUXBIN_OBQ_RESULT=${result}"
echo "FLUXBIN_OBQ_PAYLOAD=${payload}"
