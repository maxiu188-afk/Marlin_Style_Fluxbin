#!/bin/bash
# Matched extension of accepted job 6249838: only max_iters changes, 50 -> 200.
# No automatic full-model, PPL, or backend job.
#SBATCH --job-name=fluxbin2-q32-i200
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/qwen3-32b-single-linear-i200/%x-%j.out
#SBATCH --error=logs/qwen3-32b-single-linear-i200/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
snapshot="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
config="${project_root}/configs/experiments/qwen3_32b_single_linear_two_base_rank1_max_iters_200_v1.json"
baseline_result="${artifact_root}/results/qwen3-32b-single-linear-two-base-rank1-v1/fluxbin2-qwen3-32b-single-linear-6249838.json"
results_dir="${artifact_root}/results/qwen3-32b-single-linear-two-base-rank1-max-iters-200-v1"
stem="fluxbin2-qwen3-32b-single-linear-i200-${SLURM_JOB_ID}"
result="${results_dir}/${stem}.json"
payload="${results_dir}/${stem}.safetensors"
source_manifest="${results_dir}/${stem}-source-manifest.txt"
shard="model-00001-of-00017.safetensors"

mkdir -p "${results_dir}" "${project_root}/logs/qwen3-32b-single-linear-i200"
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

echo "FLUXBIN2_I200_STAGE=preflight"
for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${baseline_result}" \
    "${snapshot}/config.json" \
    "${snapshot}/model.safetensors.index.json" \
    "${snapshot}/${shard}" \
    "${project_root}/src/fluxbin_style/two_base_rank1.py" \
    "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN2_I200_MISSING=${required}" >&2
        exit 1
    fi
done
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "FLUXBIN2_I200_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

"${model_env}/bin/python" - "${config}" "${baseline_result}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baseline_path = Path(sys.argv[2])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
assert config["solver"]["max_iters"] == 200
assert baseline["solver"]["config"]["max_iters"] == 50
assert baseline["solver"]["converged_reason"] == "max_iters"
assert len(baseline["solver"]["iterations"]) == 50
assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == config["comparison"]["baseline_result_sha256"]
print("FLUXBIN2_I200_PREFLIGHT_ACCEPTED")
PY

echo "FLUXBIN2_I200_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

echo "FLUXBIN2_I200_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'comparison=accepted_job_6249838_max_iters_50_vs_max_iters_200\n'
    printf 'config=%s\n' "${config}"
    printf 'environment=%s\n' "${model_env}"
    printf 'model_revision=%s\n' "${revision}"
    printf 'snapshot=%s\n' "${snapshot}"
    printf 'baseline_result=%s\n' "${baseline_result}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    printf 'full_model_auto_launch=false\n'
    printf 'ppl_auto_launch=false\n'
    printf 'backend_auto_launch=false\n'
    sha256sum \
        "${config}" \
        "${baseline_result}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${project_root}/pyproject.toml" \
        "${project_root}/src/fluxbin_style/__init__.py" \
        "${project_root}/src/fluxbin_style/evaluation.py" \
        "${project_root}/src/fluxbin_style/two_base_rank1.py" \
        "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_single_linear_two_base_rank1_max_iters_200.sbatch"
} > "${source_manifest}"

echo "FLUXBIN2_I200_STAGE=execute"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_single_linear_two_base_rank1.py" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --baseline-result "${baseline_result}" \
    --output "${result}" \
    --payload "${payload}" \
    --source-manifest "${source_manifest}"

echo "FLUXBIN2_I200_STAGE=assert_result"
"${model_env}/bin/python" - "${result}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
comparison = result["comparison"]
assert result["status"] == "passed", result
assert result["solver"]["config"]["max_iters"] == 200, result
assert comparison["baseline_job_id"] == "6249838", result
assert comparison["baseline_max_iters"] == 50, result
assert comparison["baseline_hit_iteration_limit"], result
assert comparison["non_regressive_within_tolerance"], result
assert math.isfinite(comparison["extended_final_sse"]), result
assert result["decision"]["manual_review_required"], result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN2_I200_STRUCTURED_RESULT_ACCEPTED")
PY

echo "FLUXBIN2_I200_RESULT=${result}"
echo "FLUXBIN2_I200_PAYLOAD=${payload}"
echo "FLUXBIN2_I200_SOURCE_MANIFEST=${source_manifest}"
