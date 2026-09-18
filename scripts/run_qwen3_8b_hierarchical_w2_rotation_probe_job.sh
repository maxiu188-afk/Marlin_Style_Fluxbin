#!/usr/bin/env bash
# Run the two gated local-block probes. Never launches full-model GPTQ or PPL.
set -u

if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/run_qwen3_8b_hierarchical_w2_rotation_probe_job.sh PERSIST_ROOT NEW_JOB_DIR OUTPUT_ROOT' >&2
  exit 2
fi

rp_persist=$1
rp_job=$2
rp_output=$3
rp_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rp_python=${FLUXBIN_PYTHON:-python}
rp_policy=${FLUXBIN_EXECUTION_POLICY:-formal-a100}

case "$rp_policy" in
  formal-a100|same-device-quality) ;;
  *) echo "unsupported FLUXBIN_EXECUTION_POLICY: $rp_policy" >&2; exit 2 ;;
esac
if [ ! -d "$rp_persist" ] || [ ! -d "$rp_persist/models" ]; then
  echo "persistent volume is unavailable: $rp_persist" >&2
  exit 1
fi
if [ -e "$rp_job" ]; then
  echo "job directory already exists: $rp_job" >&2
  exit 1
fi
if [ -n "$(git -C "$rp_root" status --porcelain)" ]; then
  echo 'repository worktree is not clean; refusing a quality probe' >&2
  git -C "$rp_root" status --short >&2
  exit 1
fi
if ! command -v "$rp_python" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $rp_python" >&2
  exit 1
fi

mkdir -p "$rp_output"
mkdir "$rp_job" || exit 1
printf '%s\n' running_preflight > "$rp_job/status"
printf '%s\n' "$rp_policy" > "$rp_job/execution-policy"
git -C "$rp_root" rev-parse HEAD > "$rp_job/git-revision"
git -C "$rp_root" status --short --branch > "$rp_job/git-status"
command -v "$rp_python" > "$rp_job/python-executable.txt"
"$rp_python" -m pip freeze > "$rp_job/environment-freeze.txt"
nvidia-smi -q > "$rp_job/nvidia-smi.txt" 2>&1

export PYTHONPATH="$rp_root/src:$rp_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

rp_config="$rp_root/configs/evaluation/qwen3_8b_hierarchical_w2_rotated_v1.json"
rp_snapshot="$rp_persist/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
rp_calibration="$rp_root/artifacts/qwen3-8b-c4-v1"
rp_config_hash=$(sha256sum "$rp_config" | awk '{print $1}')

rp_fail() {
  printf '%s\n' "$2" > "$rp_job/exit-code"
  printf '%s\n' "$1" > "$rp_job/status"
  exit "$2"
}

if [ -f "$rp_output/config.sha256" ]; then
  if [ "$(awk '{print $1}' "$rp_output/config.sha256")" != "$rp_config_hash" ]; then
    rp_fail failed_mixed_config 1
  fi
else
  printf '%s  %s\n' "$rp_config_hash" "$rp_config" > "$rp_output/config.sha256"
fi
if [ -f "$rp_output/execution-policy" ]; then
  if [ "$(sed -n '1p' "$rp_output/execution-policy")" != "$rp_policy" ]; then
    rp_fail failed_mixed_execution_policy 1
  fi
else
  printf '%s\n' "$rp_policy" > "$rp_output/execution-policy"
fi

"$rp_python" -m pip check > "$rp_job/environment-check.log" 2>&1 \
  || rp_fail failed_environment_check $?

rp_common=(
  "$rp_python" -u "$rp_root/scripts/run_qwen3_8b_hierarchical_w2_rotation_probe.py"
  --config "$rp_config"
  --execution-policy "$rp_policy"
  --snapshot-root "$rp_snapshot"
  --calibration-manifest "$rp_calibration/manifest.json"
  --calibration-tokens "$rp_calibration/tokens.safetensors"
)

"${rp_common[@]}" --stage layer0 --output "$rp_job/preflight-unused.json" --validate-only \
  > "$rp_job/preflight.log" 2>&1 || rp_fail failed_probe_preflight $?
printf '%s\n' running_layer0_probe > "$rp_job/status"

rp_run_stage() {
  rp_stage=$1
  rp_result="$rp_output/$rp_stage-result.json"
  if [ ! -f "$rp_result" ]; then
    rp_command=("${rp_common[@]}" --stage "$rp_stage" --output "$rp_result")
    if [ "$rp_stage" = representative ]; then
      rp_command+=(--prior-result "$rp_output/layer0-result.json")
    fi
    printf '%q ' "${rp_command[@]}" > "$rp_job/command-$rp_stage.txt"
    printf '\n' >> "$rp_job/command-$rp_stage.txt"
    "${rp_command[@]}" > "$rp_job/$rp_stage.log" 2>&1 \
      || rp_fail "failed_${rp_stage}_probe" $?
  fi
  "$rp_python" - "$rp_result" "$rp_stage" "$rp_config_hash" "$rp_policy" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path, stage, config_hash, policy = sys.argv[1:]
result = json.loads(Path(path).read_text())
assert result["stage"] == stage, result
assert result["config_sha256"] == config_hash, result
assert result["execution"]["execution_policy"] == policy, result
assert result["full_model_launched"] is False, result
assert result["ppl_launched"] is False, result
assert result["rotated_bf16_ppl_arm"] is False, result
assert result["gate"]["action"] in {
    "advance_to_representative_probe",
    "eligible_for_manual_full_model_launch",
    "stop_without_full_model",
}, result
if stage == "layer0":
    assert result["prior_result"] is None, result
else:
    prior_path = Path(path).with_name("layer0-result.json")
    prior = json.loads(prior_path.read_text())
    prior_sha256 = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    assert result["prior_result"]["sha256"] == prior_sha256, result
    identity_fields = (
        "execution_policy", "formal_a100_device_match", "device",
        "compute_capability", "python", "torch", "transformers", "datasets",
        "safetensors", "cuda",
    )
    assert all(
        result["execution"][key] == prior["execution"][key]
        for key in identity_fields
    ), result
print("pass" if result["gate"]["passed"] else "stop")
PY
}

rp_layer0_gate=$(rp_run_stage layer0) || rp_fail failed_layer0_result_validation $?
if [ "$rp_layer0_gate" != pass ]; then
  sha256sum "$rp_output/layer0-result.json" > "$rp_job/results.sha256"
  printf '%s\n' 0 > "$rp_job/exit-code"
  printf '%s\n' completed_probe_stopped_after_layer0 > "$rp_job/status"
  exit 0
fi

printf '%s\n' running_representative_probe > "$rp_job/status"
rp_representative_gate=$(rp_run_stage representative) \
  || rp_fail failed_representative_result_validation $?
sha256sum \
  "$rp_output/layer0-result.json" \
  "$rp_output/representative-result.json" > "$rp_job/results.sha256"
printf '%s\n' 0 > "$rp_job/exit-code"
if [ "$rp_representative_gate" = pass ]; then
  printf '%s\n' completed_probe_passed_pending_manual_full_model > "$rp_job/status"
else
  printf '%s\n' completed_probe_stopped_after_representative > "$rp_job/status"
fi
