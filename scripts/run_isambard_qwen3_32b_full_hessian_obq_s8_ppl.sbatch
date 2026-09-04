#!/bin/bash
# Matched dense fake-quant PPL for independent pure/hybrid full-Hessian OBQ arms.
# No automatic distillation, backend, or performance job.
#SBATCH --job-name=fluxbin-obq-ppl-q32
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen3-32b-full-hessian-obq-s8-v3-ppl/%x-%j.out
#SBATCH --error=logs/qwen3-32b-full-hessian-obq-s8-v3-ppl/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
qbb_artifact_root="${QBB_NEW_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/qbb-new}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
revision="9216db5781bf21249d130ec9da846c4624c16137"
snapshot="${qbb_artifact_root}/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/${revision}"
config="${project_root}/configs/evaluation/qwen3_32b_wikitext2_full_hessian_obq_s8_v3.json"
artifact_id="qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized"
pure_job="6283509"
hybrid_job="6283510"
artifact_dir="${artifact_root}/artifacts/${artifact_id}"
input_results="${artifact_root}/results/${artifact_id}"
pure_result="${input_results}/pure/fluxbin-full-pure-qwen3-32b-v2-${pure_job}.json"
pure_source_manifest="${input_results}/pure/fluxbin-full-pure-qwen3-32b-v2-${pure_job}-source-manifest.txt"
hybrid_result="${input_results}/hybrid_s8/fluxbin-full-hybrid_s8-qwen3-32b-v2-${hybrid_job}.json"
hybrid_source_manifest="${input_results}/hybrid_s8/fluxbin-full-hybrid_s8-qwen3-32b-v2-${hybrid_job}-source-manifest.txt"
protocol_dir="${qbb_artifact_root}/protocols/qwen3-wikitext2-v1"
protocol_manifest="${protocol_dir}/qbbnew-qwen3-wikitext2-protocol-6109964.json"
token_artifact="${protocol_dir}/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
bf16_reference="${qbb_artifact_root}/results/qwen3-32b-shared-column-joint-ppl/qbbnew-qwen3-32b-shared-column-joint-ppl-6154681.json"
results_dir="${artifact_root}/results/qwen3-32b-full-hessian-obq-s8-v3-ppl"
stem="fluxbin-qwen3-32b-full-hessian-obq-s8-v3-ppl-${SLURM_JOB_ID}"
result="${results_dir}/${stem}.json"
source_manifest="${results_dir}/${stem}-source-manifest.txt"

mkdir -p "${results_dir}" "${project_root}/logs/qwen3-32b-full-hessian-obq-s8-v3-ppl"
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
export PYTHONPATH="${project_root}/src:${project_root}/scripts"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "FLUXBIN_OBQ_PPL_STAGE=preflight"
for required in \
    "${model_env}/bin/python" \
    "${config}" \
    "${snapshot}/config.json" \
    "${snapshot}/model.safetensors.index.json" \
    "${pure_result}" \
    "${pure_source_manifest}" \
    "${artifact_dir}/pure" \
    "${hybrid_result}" \
    "${hybrid_source_manifest}" \
    "${artifact_dir}/hybrid_s8" \
    "${protocol_manifest}" \
    "${token_artifact}" \
    "${bf16_reference}" \
    "${project_root}/src/fluxbin_style/evaluation.py" \
    "${project_root}/scripts/run_qwen3_two_base_rank1_s8_ppl.py" \
    "${project_root}/scripts/run_qwen3_full_hessian_obq_s8_ppl.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_OBQ_PPL_MISSING=${required}" >&2
        exit 1
    fi
done
for output in "${result}" "${source_manifest}"; do
    if [[ -e "${output}" ]]; then
        echo "FLUXBIN_OBQ_PPL_REFUSES_OVERWRITE=${output}" >&2
        exit 1
    fi
done
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "FLUXBIN_OBQ_PPL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

"${model_env}/bin/python" - "${config}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert config["schema_version"] == 2, config
assert config["model"]["expected_hidden_layers"] == 64, config
assert config["model"]["expected_tensor_count"] == 448, config
assert config["model"]["expected_parameter_count"] == 31_205_621_760, config
assert config["decomposition"]["artifact_id"] == (
    "qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized"
), config
assert not config["decomposition"]["arms_share_global_payload"], config
assert config["decomposition"]["arms"]["pure"]["job_id"] == "6283509", config
assert config["decomposition"]["arms"]["hybrid_s8"]["job_id"] == "6283510", config
assert config["evaluation"]["arms"] == ["bf16", "pure", "hybrid_s8"], config
assert config["accepted_protocol"]["full_block_count"] == 146, config
assert config["accepted_protocol"]["scored_transition_count"] == 298_862, config
assert config["accepted_bf16_reference"]["perplexity"] == 7.6108390395557874, config
assert not config["acceptance"]["backend_auto_launch"], config
assert not config["acceptance"]["distillation_auto_launch"], config
print("FLUXBIN_OBQ_PPL_CONFIG_ACCEPTED")
PY

"${model_env}/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("FLUXBIN_OBQ_PPL_RUNTIME_ACCEPTED")
PY

echo "FLUXBIN_OBQ_PPL_STAGE=tests"
"${model_env}/bin/python" -m unittest discover -s tests -v

mapfile -t model_shards < <(
    "${model_env}/bin/python" - "${snapshot}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
assert index["metadata"]["total_size"] == 65_524_246_528
for name in sorted(set(index["weight_map"].values())):
    path = root / name
    assert path.is_file(), path
    print(path)
PY
)
if [[ "${#model_shards[@]}" -ne 17 ]]; then
    echo "FLUXBIN_OBQ_PPL_SHARD_COUNT_DRIFTED=${#model_shards[@]}" >&2
    exit 1
fi

echo "FLUXBIN_OBQ_PPL_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'environment=%s\n' "${model_env}"
    printf 'model_revision=%s\n' "${revision}"
    printf 'artifact_id=%s\n' "${artifact_id}"
    printf 'pure_job=%s\n' "${pure_job}"
    printf 'hybrid_job=%s\n' "${hybrid_job}"
    printf 'evaluation_arms=bf16,pure,hybrid_s8\n'
    printf 'backend_auto_launch=false\n'
    printf 'distillation_auto_launch=false\n'
    sha256sum \
        "${config}" \
        "${snapshot}/config.json" \
        "${snapshot}/model.safetensors.index.json" \
        "${model_shards[@]}" \
        "${pure_result}" \
        "${pure_source_manifest}" \
        "${hybrid_result}" \
        "${hybrid_source_manifest}" \
        "${protocol_manifest}" \
        "${token_artifact}" \
        "${bf16_reference}" \
        "${project_root}/pyproject.toml" \
        "${project_root}/src/fluxbin_style/evaluation.py" \
        "${project_root}/scripts/run_qwen3_two_base_rank1_s8_ppl.py" \
        "${project_root}/scripts/run_qwen3_full_hessian_obq_s8_ppl.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_full_hessian_obq_s8_ppl.sbatch"
} > "${source_manifest}"

echo "FLUXBIN_OBQ_PPL_STAGE=execute"
"${model_env}/bin/python" "${project_root}/scripts/run_qwen3_full_hessian_obq_s8_ppl.py" \
    --config "${config}" \
    --snapshot-root "${snapshot}" \
    --protocol-manifest "${protocol_manifest}" \
    --token-artifact "${token_artifact}" \
    --pure-result "${pure_result}" \
    --pure-source-manifest "${pure_source_manifest}" \
    --pure-artifact-dir "${artifact_dir}/pure" \
    --hybrid-result "${hybrid_result}" \
    --hybrid-source-manifest "${hybrid_source_manifest}" \
    --hybrid-artifact-dir "${artifact_dir}/hybrid_s8" \
    --accepted-bf16-result "${bf16_reference}" \
    --source-manifest "${source_manifest}" \
    --output "${result}"

echo "FLUXBIN_OBQ_PPL_STAGE=assert_result"
"${model_env}/bin/python" - "${result}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] in ("completed_pending_review", "completed_invalid_metrics"), result
assert result["runtime"]["device_name"] == "NVIDIA GH200 120GB", result
assert result["protocol"]["exact_accepted_artifact_reused"], result
assert result["protocol"]["scored_transition_count"] == 298_862, result
for arm in ("pure", "hybrid_s8"):
    ledger = result["decomposition_ledgers"][arm]["aggregate"]
    assert ledger["layer_count"] == 64, result
    assert ledger["tensor_count"] == 448, result
    assert ledger["parameter_count"] == 31_205_621_760, result
    materialization = result["materialization"][arm]
    assert materialization["layer_count"] == 64, result
    assert materialization["tensor_count"] == 448, result
    assert materialization["parameter_count"] == 31_205_621_760, result
    assert materialization["all_payload_and_internal_tensor_hashes_validated"], result
for arm in ("bf16", "pure", "hybrid_s8"):
    assert result["arms"][arm]["scored_transition_count"] == 298_862, result
if result["status"] == "completed_pending_review":
    assert result["acceptance_checks"]["all_metrics_finite"], result
    assert result["acceptance_checks"]["accepted_bf16_reproduced_within_tolerance"], result
    assert result["acceptance_checks"]["all_layer_payload_and_internal_tensor_hashes_validated"], result
    assert result["acceptance_checks"]["complete_arm_coverage"], result
    for arm in ("bf16", "pure", "hybrid_s8"):
        assert result["arms"][arm]["metrics_valid"], result
        assert math.isfinite(result["arms"][arm]["perplexity"]), result
assert result["next_stage"] == "not_launched", result
print("FLUXBIN_OBQ_PPL_STRUCTURED_RESULT_READY_FOR_REVIEW")
PY

echo "FLUXBIN_OBQ_PPL_RESULT=${result}"
echo "FLUXBIN_OBQ_PPL_SOURCE_MANIFEST=${source_manifest}"
