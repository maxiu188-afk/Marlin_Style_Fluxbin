#!/bin/bash
# One real Qwen3-32B Linear only. No automatic full-model or PPL job.
# Retains the accepted pure two-base arm and adds sparse s=8 residual refinement.
#SBATCH --job-name=fluxbin2-s8-q32
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen3-32b-single-linear-s8/%x-%j.out
#SBATCH --error=logs/qwen3-32b-single-linear-s8/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
snapshot="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
config="${project_root}/configs/experiments/qwen3_32b_single_linear_two_base_rank1_s8_v1.json"
global_results_dir="${artifact_root}/results/qwen3-32b-single-linear-two-base-rank1-v1"
global_reference_result="${global_results_dir}/fluxbin2-qwen3-32b-single-linear-6249838.json"
results_dir="${artifact_root}/results/qwen3-32b-single-linear-two-base-rank1-s8-v1"
result="${results_dir}/fluxbin2-s8-qwen3-32b-single-linear-${SLURM_JOB_ID}.json"
payload="${results_dir}/fluxbin2-s8-qwen3-32b-single-linear-${SLURM_JOB_ID}.safetensors"
source_manifest="${results_dir}/fluxbin2-s8-qwen3-32b-single-linear-${SLURM_JOB_ID}-source-manifest.txt"
shard="model-00001-of-00017.safetensors"

mkdir -p "${results_dir}" "${project_root}/logs/qwen3-32b-single-linear-s8"
task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/fluxbin2-s8-q32-${SLURM_JOB_ID}"
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

echo "FLUXBIN2_S8_SINGLE_STAGE=preflight"
for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${global_reference_result}" \
    "${snapshot}/config.json" \
    "${snapshot}/model.safetensors.index.json" \
    "${snapshot}/${shard}" \
    "${project_root}/src/fluxbin_style/two_base_rank1.py" \
    "${project_root}/src/fluxbin_style/residual_refinement.py" \
    "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1_s8.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN2_S8_SINGLE_MISSING=${required}" >&2
        exit 1
    fi
done
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "FLUXBIN2_S8_SINGLE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

"${model_env}/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("FLUXBIN2_S8_SINGLE_RUNTIME_ACCEPTED")
PY

echo "FLUXBIN2_S8_SINGLE_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

echo "FLUXBIN2_S8_SINGLE_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'config=%s\n' "${config}"
    printf 'environment=%s\n' "${model_env}"
    printf 'model_revision=%s\n' "${revision}"
    printf 'snapshot=%s\n' "${snapshot}"
    printf 'shard_symlink=%s\n' "$(readlink "${snapshot}/${shard}")"
    printf 'global_reference_result=%s\n' "${global_reference_result}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${config}" \
        "${global_reference_result}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${project_root}/pyproject.toml" \
        "${project_root}/src/fluxbin_style/__init__.py" \
        "${project_root}/src/fluxbin_style/evaluation.py" \
        "${project_root}/src/fluxbin_style/two_base_rank1.py" \
        "${project_root}/src/fluxbin_style/residual_refinement.py" \
        "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1_s8.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_single_linear_two_base_rank1_s8.sbatch"
} > "${source_manifest}"

echo "FLUXBIN2_S8_SINGLE_STAGE=execute"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1_s8.py" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --global-reference-result "${global_reference_result}" \
    --output "${result}" \
    --payload "${payload}" \
    --source-manifest "${source_manifest}"

echo "FLUXBIN2_S8_SINGLE_STAGE=assert_result"
"${model_env}/bin/python" - "${result}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["contract"]["global_num_bases"] == 2, result
assert result["contract"]["global_group_size"] == 128, result
assert result["contract"]["residual_columns_per_group"] == 8, result
assert result["contract"]["refinement_num_bases"] == 2, result
assert not result["contract"]["hessian_error_propagation"], result
assert result["global_reference"]["job_id"] == "6249838", result
assert result["global_reference"]["validated"], result
global_sse = result["reconstruction"]["global_two_base_rank_one"]["squared_error"]
hybrid_sse = result["reconstruction"]["hybrid_s8"]["squared_error"]
assert math.isfinite(global_sse) and math.isfinite(hybrid_sse), result
assert hybrid_sse < global_sse, result
assert result["reconstruction"]["non_selected_max_abs_delta_vs_global"] == 0.0, result
assert result["selection"]["indices_unique_within_group"], result
assert result["decision"]["manual_review_required"], result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN2_S8_SINGLE_STRUCTURED_RESULT_ACCEPTED")
PY

echo "FLUXBIN2_S8_SINGLE_RESULT=${result}"
echo "FLUXBIN2_S8_SINGLE_PAYLOAD=${payload}"
echo "FLUXBIN2_S8_SINGLE_SOURCE_MANIFEST=${source_manifest}"
