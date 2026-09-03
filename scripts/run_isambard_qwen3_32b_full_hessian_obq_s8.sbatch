#!/bin/bash
# One independently propagated full-model arm. Resumable by completed layer.
#SBATCH --job-name=fluxbin-full-obq
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen3-32b-full-hessian-obq-s8-v2/%x-%j.out
#SBATCH --error=logs/qwen3-32b-full-hessian-obq-s8-v2/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

arm="${FLUXBIN_OBQ_ARM:?set FLUXBIN_OBQ_ARM to pure or hybrid_s8}"
if [[ "${arm}" != "pure" && "${arm}" != "hybrid_s8" ]]; then
    echo "FLUXBIN_FULL_INVALID_ARM=${arm}" >&2
    exit 1
fi

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
single_dir="${artifact_root}/results/qwen3-32b-single-linear-hessian-obq-s8-v2"
single_result="${single_dir}/fluxbin-obq-qwen3-32b-single-linear-s8-v2-6271396.json"
single_payload="${single_dir}/fluxbin-obq-qwen3-32b-single-linear-s8-v2-6271396.safetensors"
config="${project_root}/configs/experiments/qwen3_32b_full_hessian_obq_s8_v2.json"
artifact_id="qwen3-32b-full-hessian-obq-s8-v2"
arm_artifact_dir="${artifact_root}/artifacts/${artifact_id}/${arm}"
results_dir="${artifact_root}/results/${artifact_id}/${arm}"
stem="fluxbin-full-${arm}-qwen3-32b-v2-${SLURM_JOB_ID}"
result="${results_dir}/${stem}.json"
source_manifest="${results_dir}/${stem}-source-manifest.txt"

mkdir -p \
    "${arm_artifact_dir}" \
    "${results_dir}" \
    "${project_root}/logs/qwen3-32b-full-hessian-obq-s8-v2"
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
    "${single_result}" \
    "${single_payload}" \
    "${project_root}/src/fluxbin_style/hessian_obq.py" \
    "${project_root}/src/fluxbin_style/qwen3_sequential.py" \
    "${project_root}/scripts/run_qwen3_full_hessian_obq_s8.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_FULL_MISSING=${required}" >&2
        exit 1
    fi
done
for path in "${result}" "${source_manifest}"; do
    if [[ -e "${path}" ]]; then
        echo "FLUXBIN_FULL_REFUSES_OVERWRITE=${path}" >&2
        exit 1
    fi
done
if [[ -n "$(git status --short)" ]]; then
    echo "FLUXBIN_FULL_DIRTY_CHECKOUT" >&2
    git status --short >&2
    exit 1
fi

available_bytes=$(df -PB1 "${artifact_root}" | awk 'NR==2 {print $4}')
if (( available_bytes < 30000000000 )); then
    echo "FLUXBIN_FULL_INSUFFICIENT_SPACE=${available_bytes}" >&2
    exit 1
fi

echo "FLUXBIN_FULL_STAGE=tests arm=${arm}"
"${model_env}/bin/python" -m unittest discover -s tests -v

mapfile -t shard_paths < <(
    "${model_env}/bin/python" - "${snapshot}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
for shard in sorted(set(index["weight_map"].values())):
    path = root / shard
    assert path.is_file(), path
    print(path)
PY
)
if [[ "${#shard_paths[@]}" -ne 17 ]]; then
    echo "FLUXBIN_FULL_SHARD_COUNT_DRIFTED=${#shard_paths[@]}" >&2
    exit 1
fi

echo "FLUXBIN_FULL_STAGE=source_manifest arm=${arm}"
{
    printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=clean\n'
    printf 'arm=%s\n' "${arm}"
    printf 'model_revision=%s\n' "${revision}"
    printf 'artifact_id=%s\n' "${artifact_id}"
    printf 'artifact_dir=%s\n' "${arm_artifact_dir}"
    printf 'calibration_job=6271395\n'
    printf 'single_linear_gate_job=6271396\n'
    printf 'ppl_auto_launch=false\n'
    printf 'backend_auto_launch=false\n'
    sha256sum \
        "${config}" \
        "${calibration_manifest}" \
        "${calibration_tokens}" \
        "${single_result}" \
        "${single_payload}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${shard_paths[@]}" \
        "${project_root}/src/fluxbin_style/__init__.py" \
        "${project_root}/src/fluxbin_style/evaluation.py" \
        "${project_root}/src/fluxbin_style/hessian_obq.py" \
        "${project_root}/src/fluxbin_style/packing.py" \
        "${project_root}/src/fluxbin_style/qwen3.py" \
        "${project_root}/src/fluxbin_style/qwen3_sequential.py" \
        "${project_root}/src/fluxbin_style/residual_refinement.py" \
        "${project_root}/src/fluxbin_style/two_base_rank1.py" \
        "${project_root}/scripts/run_qwen3_full_hessian_obq_s8.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_full_hessian_obq_s8.sbatch"
} > "${source_manifest}"

echo "FLUXBIN_FULL_STAGE=execute arm=${arm}"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_full_hessian_obq_s8.py" \
    --arm "${arm}" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --calibration-manifest "${calibration_manifest}" \
    --calibration-tokens "${calibration_tokens}" \
    --single-linear-result "${single_result}" \
    --single-linear-payload "${single_payload}" \
    --artifact-dir "${arm_artifact_dir}" \
    --source-manifest "${source_manifest}" \
    --project-root "${project_root}" \
    --output "${result}"

"${model_env}/bin/python" - "${result}" "${arm}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arm = sys.argv[2]
assert result["status"] == "completed_pending_review", result
assert result["arm"] == arm, result
assert result["aggregate"]["layer_count"] == 64, result
assert result["aggregate"]["tensor_count"] == 448, result
assert result["aggregate"]["parameter_count"] == 31_205_621_760, result
assert math.isfinite(result["aggregate"]["squared_error"]), result
assert math.isfinite(result["aggregate"]["calibration_total_output_squared_error"]), result
assert len(result["artifacts"]) == 64, result
assert result["decision"]["ppl_auto_launch"] is False, result
assert result["decision"]["backend_auto_launch"] is False, result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN_FULL_STRUCTURED_RESULT_READY_FOR_REVIEW")
PY

echo "FLUXBIN_FULL_RESULT=${result}"
echo "FLUXBIN_FULL_ARTIFACT_DIR=${arm_artifact_dir}"
