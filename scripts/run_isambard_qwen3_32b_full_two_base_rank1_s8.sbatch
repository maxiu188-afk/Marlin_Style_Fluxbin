#!/bin/bash
# Full 448-Linear Qwen3-32B reconstruction. No automatic PPL or backend job.
# One shared global two-base arm plus sparse s=8 residual refinement.
#SBATCH --job-name=fluxbin-full-q32-s8
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3-32b-full-two-base-rank1-s8-v1/%x-%j.out
#SBATCH --error=logs/qwen3-32b-full-two-base-rank1-s8-v1/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
snapshot="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
config="${project_root}/configs/experiments/qwen3_32b_full_two_base_rank1_s8_v1.json"
single_result_dir="${artifact_root}/results/qwen3-32b-single-linear-two-base-rank1-s8-v1"
single_result="${single_result_dir}/fluxbin2-s8-qwen3-32b-single-linear-6250990.json"
artifact_id="qwen3-32b-full-two-base-rank1-s8-v1"
decomposition_dir="${artifact_root}/artifacts/${artifact_id}/decompositions"
results_dir="${artifact_root}/results/${artifact_id}"
stem="fluxbin-qwen3-32b-full-two-base-rank1-s8-v1-${SLURM_JOB_ID}"
result="${results_dir}/${stem}.json"
source_manifest="${results_dir}/${stem}-source-manifest.txt"

mkdir -p \
    "${decomposition_dir}" \
    "${results_dir}" \
    "${project_root}/logs/qwen3-32b-full-two-base-rank1-s8-v1"
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

echo "FLUXBIN_FULL_STAGE=preflight"
for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${single_result}" \
    "${snapshot}/config.json" \
    "${snapshot}/model.safetensors.index.json" \
    "${project_root}/src/fluxbin_style/evaluation.py" \
    "${project_root}/src/fluxbin_style/packing.py" \
    "${project_root}/src/fluxbin_style/qwen3.py" \
    "${project_root}/src/fluxbin_style/residual_refinement.py" \
    "${project_root}/src/fluxbin_style/two_base_rank1.py" \
    "${project_root}/scripts/run_qwen3_full_two_base_rank1_s8.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_FULL_MISSING=${required}" >&2
        exit 1
    fi
done
for output in "${result}" "${source_manifest}"; do
    if [[ -e "${output}" ]]; then
        echo "FLUXBIN_FULL_REFUSES_OVERWRITE=${output}" >&2
        exit 1
    fi
done
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "FLUXBIN_FULL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

"${model_env}/bin/python" - "${config}" "${artifact_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = config["model"]
assert model["repo_id"] == "Qwen/Qwen3-32B", config
assert model["revision"] == "9216db5781bf21249d130ec9da846c4624c16137", config
assert model["expected_hidden_layers"] == 64, config
assert model["expected_tensor_count"] == 448, config
assert model["expected_parameter_count"] == 31_205_621_760, config
assert model["expected_weight_shards"] == 17, config
assert model["expected_checkpoint_bytes"] == 65_524_246_528, config
assert config["algorithm"]["global_num_bases"] == 2, config
assert config["algorithm"]["global_group_size"] == 128, config
assert config["algorithm"]["residual_columns_per_group"] == 8, config
assert config["algorithm"]["refinement_num_bases"] == 2, config
assert config["single_linear_gate"]["outcome"] == "go_full_model", config
assert config["artifact_policy"]["share_global_arm_between_pure_and_hybrid"], config
assert config["artifact_policy"]["resumable"], config
assert not config["decision"]["ppl_auto_launch"], config
assert not config["decision"]["backend_auto_launch"], config
stat = os.statvfs(Path(sys.argv[2]))
available_bytes = stat.f_bavail * stat.f_frsize
required_free_bytes = 30_000_000_000
if available_bytes < required_free_bytes:
    raise RuntimeError(f"insufficient free space: {available_bytes} < {required_free_bytes}")
print(f"FLUXBIN_FULL_AVAILABLE_BYTES={available_bytes}")
print("FLUXBIN_FULL_CONFIG_ACCEPTED")
PY

"${model_env}/bin/python" - <<'PY'
import torch
from fluxbin_style import pack_two_bases, unpack_two_bases

assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
torch.manual_seed(20260902)
bases = torch.where(
    torch.rand(2, 16, 128, device="cuda:0") < 0.5,
    torch.tensor(-1, dtype=torch.int8, device="cuda:0"),
    torch.tensor(1, dtype=torch.int8, device="cuda:0"),
)
assert torch.equal(unpack_two_bases(pack_two_bases(bases)), bases)
print("FLUXBIN_FULL_RUNTIME_ACCEPTED")
PY

echo "FLUXBIN_FULL_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

mapfile -t shard_paths < <(
    "${model_env}/bin/python" - "${snapshot}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
assert index["metadata"]["total_size"] == 65_524_246_528
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

echo "FLUXBIN_FULL_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'environment=%s\n' "${model_env}"
    printf 'model_revision=%s\n' "${revision}"
    printf 'snapshot=%s\n' "${snapshot}"
    printf 'single_linear_gate_job=6250990\n'
    printf 'artifact_id=%s\n' "${artifact_id}"
    printf 'decomposition_dir=%s\n' "${decomposition_dir}"
    printf 'stored_arms=global_two_base_rank_one,sparse_residual_refinement_s8\n'
    printf 'global_arm_shared=true\n'
    printf 'payload_format=fluxbin-two-base-interleaved-2bit-v1\n'
    printf 'ppl_auto_launch=false\n'
    printf 'backend_auto_launch=false\n'
    sha256sum \
        "${config}" \
        "${single_result}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${shard_paths[@]}" \
        "${project_root}/pyproject.toml" \
        "${project_root}/src/fluxbin_style/__init__.py" \
        "${project_root}/src/fluxbin_style/evaluation.py" \
        "${project_root}/src/fluxbin_style/packing.py" \
        "${project_root}/src/fluxbin_style/qwen3.py" \
        "${project_root}/src/fluxbin_style/residual_refinement.py" \
        "${project_root}/src/fluxbin_style/two_base_rank1.py" \
        "${project_root}/scripts/run_qwen3_full_two_base_rank1_s8.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_full_two_base_rank1_s8.sbatch"
} > "${source_manifest}"

echo "FLUXBIN_FULL_STAGE=execute"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_full_two_base_rank1_s8.py" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --single-linear-result "${single_result}" \
    --artifact-dir "${decomposition_dir}" \
    --output "${result}" \
    --source-manifest "${source_manifest}" \
    --project-root "${project_root}"

echo "FLUXBIN_FULL_STAGE=assert_result"
"${model_env}/bin/python" - "${result}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["execution"]["tensor_count"] == 448, result
assert result["execution"]["artifact_file_count"] == 896, result
assert result["aggregate"]["parameter_count"] == 31_205_621_760, result
assert result["aggregate"]["every_tensor_strictly_improved"], result
global_sse = result["aggregate"]["global_two_base_rank_one"]["squared_error"]
hybrid_sse = result["aggregate"]["hybrid_s8"]["squared_error"]
assert math.isfinite(global_sse) and math.isfinite(hybrid_sse), result
assert hybrid_sse < global_sse, result
assert result["contract"]["global_num_bases"] == 2, result
assert result["contract"]["residual_columns_per_group"] == 8, result
assert result["contract"]["stored_arms"] == [
    "global_two_base_rank_one",
    "sparse_residual_refinement_s8",
], result
assert result["contract"]["global_arm_shared_by_pure_and_hybrid"], result
assert result["storage"]["total_payload_bytes"] > 0, result
assert len(result["tensors"]) == 448, result
assert not result["decision"]["ppl_auto_launch"], result
assert not result["decision"]["backend_auto_launch"], result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN_FULL_STRUCTURED_RESULT_READY_FOR_REVIEW")
PY

echo "FLUXBIN_FULL_RESULT=${result}"
echo "FLUXBIN_FULL_ARTIFACT_DIR=${decomposition_dir}"
echo "FLUXBIN_FULL_SOURCE_MANIFEST=${source_manifest}"
