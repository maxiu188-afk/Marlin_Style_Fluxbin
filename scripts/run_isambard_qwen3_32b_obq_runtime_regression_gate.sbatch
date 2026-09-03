#!/bin/bash
# CPU-only gate: continuation jobs depend on exact old/new layer comparisons.
#SBATCH --job-name=fluxbin-obq-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/qwen3-32b-full-hessian-obq-s8-v3/%x-%j.out
#SBATCH --error=logs/qwen3-32b-full-hessian-obq-s8-v3/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7

pure_job="${FLUXBIN_PURE_REGRESSION_JOB:?set the pure regression job id}"
hybrid_job="${FLUXBIN_HYBRID_REGRESSION_JOB:?set the hybrid regression job id}"
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
model_env="${QBB_MODEL_ENV:-${PROJECTDIR}/${USER}/qbb-envs/qbb-model-v1}"
artifact_root="${FLUXBIN_STYLE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/marlin-style-fluxbin}"
reference_id="qwen3-32b-full-hessian-obq-s8-v2"
candidate_id="qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized"
reference_root="${artifact_root}/artifacts/${reference_id}"
candidate_root="${artifact_root}/artifacts/${candidate_id}"
candidate_results="${artifact_root}/results/${candidate_id}"
gate_dir="${candidate_results}/regression-gate"
pure_result="${candidate_results}/pure/fluxbin-full-pure-qwen3-32b-v2-${pure_job}.json"
hybrid_result="${candidate_results}/hybrid_s8/fluxbin-full-hybrid_s8-qwen3-32b-v2-${hybrid_job}.json"
pure_comparison="${gate_dir}/pure-0-10-${SLURM_JOB_ID}.json"
hybrid_comparison="${gate_dir}/hybrid-s8-0-5-${SLURM_JOB_ID}.json"
gate_result="${gate_dir}/runtime-regression-gate-${SLURM_JOB_ID}.json"
source_manifest="${gate_dir}/runtime-regression-gate-${SLURM_JOB_ID}-source-manifest.txt"

mkdir -p "${gate_dir}" "${project_root}/logs/qwen3-32b-full-hessian-obq-s8-v3"
for required in \
    "${model_env}/bin/python" \
    "${pure_result}" \
    "${hybrid_result}" \
    "${project_root}/scripts/compare_full_hessian_obq_artifacts.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "FLUXBIN_REGRESSION_GATE_MISSING=${required}" >&2
        exit 1
    fi
done
for path in \
    "${pure_comparison}" \
    "${hybrid_comparison}" \
    "${gate_result}" \
    "${source_manifest}"; do
    if [[ -e "${path}" ]]; then
        echo "FLUXBIN_REGRESSION_GATE_REFUSES_OVERWRITE=${path}" >&2
        exit 1
    fi
done
if [[ -n "$(git status --short)" ]]; then
    echo "FLUXBIN_REGRESSION_GATE_DIRTY_CHECKOUT" >&2
    git status --short >&2
    exit 1
fi

{
    printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=clean\n'
    printf 'pure_regression_job=%s\n' "${pure_job}"
    printf 'hybrid_regression_job=%s\n' "${hybrid_job}"
    printf 'reference_artifact_id=%s\n' "${reference_id}"
    printf 'candidate_artifact_id=%s\n' "${candidate_id}"
    sha256sum \
        "${pure_result}" \
        "${hybrid_result}" \
        "${project_root}/scripts/compare_full_hessian_obq_artifacts.py" \
        "${project_root}/scripts/run_isambard_qwen3_32b_obq_runtime_regression_gate.sbatch"
} > "${source_manifest}"

export PYTHONPATH="${project_root}/src"
"${model_env}/bin/python" "${project_root}/scripts/compare_full_hessian_obq_artifacts.py" \
    --arm pure \
    --reference-root "${reference_root}/pure" \
    --candidate-root "${candidate_root}/pure" \
    --last-layer 10 \
    --output "${pure_comparison}"
"${model_env}/bin/python" "${project_root}/scripts/compare_full_hessian_obq_artifacts.py" \
    --arm hybrid_s8 \
    --reference-root "${reference_root}/hybrid_s8" \
    --candidate-root "${candidate_root}/hybrid_s8" \
    --last-layer 5 \
    --output "${hybrid_comparison}"

"${model_env}/bin/python" - \
    "${pure_result}" \
    "${hybrid_result}" \
    "${pure_comparison}" \
    "${hybrid_comparison}" \
    "${source_manifest}" \
    "${gate_result}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

from fluxbin_style import atomic_json, sha256_file

pure_result_path, hybrid_result_path, pure_path, hybrid_path, manifest, output = map(
    Path,
    sys.argv[1:],
)
pure_result = json.loads(pure_result_path.read_text(encoding="utf-8"))
hybrid_result = json.loads(hybrid_result_path.read_text(encoding="utf-8"))
pure = json.loads(pure_path.read_text(encoding="utf-8"))
hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
assert pure_result["status"] == "partial_completed_pending_review", pure_result
assert pure_result["aggregate"]["layer_count"] == 11, pure_result
assert pure_result["execution"]["newly_quantized_layers"] == 11, pure_result
assert hybrid_result["status"] == "partial_completed_pending_review", hybrid_result
assert hybrid_result["aggregate"]["layer_count"] == 6, hybrid_result
assert hybrid_result["execution"]["newly_quantized_layers"] == 6, hybrid_result
assert pure["status"] == hybrid["status"] == "passed"
assert pure["all_payloads_bit_exact"] and hybrid["all_payloads_bit_exact"]
atomic_json(
    output,
    {
        "schema_version": 1,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pure_regression_result_sha256": sha256_file(pure_result_path),
        "hybrid_regression_result_sha256": sha256_file(hybrid_result_path),
        "pure_comparison_sha256": sha256_file(pure_path),
        "hybrid_comparison_sha256": sha256_file(hybrid_path),
        "source_manifest_sha256": sha256_file(manifest),
        "continuation_authorized": True,
        "ppl_auto_launch": False,
        "backend_auto_launch": False,
    },
)
print("FLUXBIN_RUNTIME_REGRESSION_GATE=passed")
print(f"FLUXBIN_RUNTIME_REGRESSION_GATE_RESULT={output}")
PY
