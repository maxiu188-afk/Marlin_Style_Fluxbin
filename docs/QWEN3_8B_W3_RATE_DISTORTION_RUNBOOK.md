# Qwen3-8B uniform W3 vs QBB rate--distortion runbook

Status: W3 and QBB FP16-scale artifacts completed on A100 PCIe on 2026-09-15;
the full four-arm input/hash preflight passed, but PPL has not been launched.
The next available GPU is an RTX PRO 4500 Blackwell, so Stage 3 has an explicit
same-device quality-comparison mode in addition to the unchanged formal A100
mode. See `QWEN3_8B_W3_RATE_DISTORTION_STATUS.md` before resuming at Stage 3.

## Question and frozen boundary

This experiment asks whether standard uniform symmetric GPTQ W3 g128 already
dominates the current distilled two-base plus salient-8 QBB representation at a
similar storage rate. It does not add a CUDA kernel, change an inference
backend, benchmark speed, retrain QBB, or tune several GPTQ configurations.

All four arms are scored by the existing dense-BF16 WikiText-2 evaluator:

- exact model: `Qwen/Qwen3-8B` revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- 36 transformer layers and exactly 252 block Linear weights;
- 146 independent 2048-token test blocks;
- 298,862 scored next-token positions;
- BF16 model/logits, FP32 cross entropy, SDPA, batch 1, no cache, no TF32;
- embeddings and `lm_head` remain at the pinned BF16 source weights.

The frozen experiment config is
`configs/evaluation/qwen3_8b_w3_rate_distortion_v1.json`.

## Prepared arms

1. `bf16`: retain the accepted PPL `9.724944980689296` as the reference, reload
   the pinned source snapshot, and reproduce it once as an evaluator-drift gate
   (absolute tolerance `1e-6`). No BF16 model artifact is regenerated.
2. `qbb_current`: reload the accepted step-400 QBB artifact, decode all 252
   weights to BF16, and score it second. Its accepted PPL
   `13.169788494766266` is also a `1e-6` reproduction gate.
3. `gptq_w3_g128_sym`: GPTQModel 7.4.0, 3 bits, group size 128, symmetric,
   `desc_act=true`, `act_group_aware=false`, `static_groups=false`,
   `true_sequential=true`, no MSE grid search, no fallback, and no
   embedding/`lm_head` quantization. It reuses the exact accepted
   256 x 2048 C4 token artifact without sorting, concatenation or resampling.
4. `qbb_fp16_scales`: cast the four QBB scale families from FP32 to FP16;
   packed signs and salient indices must remain bit-identical. There is no
   optimization, training or distillation.

GPTQModel is not part of the PPL evaluator. The W3 runner saves the packed raw
artifact, reloads it with the `GPTQ_TORCH` backend, explicitly forces
`GPTQ_TORCH_TRITON_DEQUANT=0`, and writes exactly 252 decoded BF16 weights. The
frozen repository scorer then copies those weights into the pinned source
model. This avoids accidentally entering an optional 3-bit Triton
dequantization path merely because Triton is installed in the server image.

## Static storage audit before execution

The 252 target weights contain exactly `6,945,767,424` scalar weights.

| Representation | Persistent tensor bytes | Analytical effective bits/weight | Notes |
|---|---:|---:|---|
| BF16 target Linears | 13,891,534,848 | 16.000000 | target Linear weights only |
| current QBB, FP32 scales | 2,724,636,672 | 3.138184 | seven source payload tensors; no legacy lookup |
| QBB, FP16 scales | 2,284,886,016 | 2.631687 | same signs and indices |
| symmetric GPTQ W3 g128 semantic core | 2,713,190,400 | 3.125000 | 3-bit codes plus one FP16 scale/group |

The current QBB source safetensors files are `75,689,608` bytes per layer, or
`2,724,825,888` bytes in total (`3.138402` bits/weight including their file
headers). The older approximate `3.1458` number is therefore not the exact
deployment rate once the unused lookup is excluded.

GPTQ's final effective rate must not be assumed to be 3.125. The formal result
uses the actual packed `qweight`, `scales`, `qzeros` and `g_idx` tensor bytes
found after serialization. Padding, stored constant zero points and activation
ordering metadata are broken out in `storage.json`. The raw GPTQModel
checkpoint also records its whole-container byte size, but that number includes
unchanged embeddings, `lm_head` and normalization tensors and is therefore not
used as the 252-Linear quantization rate.

## Server paths

After attaching the retained network volume, define task-specific paths. Do not
substitute a different snapshot or recreate either token artifact.

```bash
project_root=/workspace/repos/marlin-style-fluxbin
snapshot_root=/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
calibration_root=/workspace/repos/marlin-style-fluxbin/artifacts/qwen3-8b-c4-v1
protocol_root=/workspace/data/qwen3-wikitext2-v1
qbb_root=/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1
gptq_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1
qbb_fp16_root=/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-fp16-scales-v1
result_root=/workspace/artifacts/qwen3_8b_w3_rate_distortion
```

The exact calibration filenames must be read from the accepted calibration
manifest. The known test files are:

```bash
protocol_manifest="$protocol_root/qbbnew-qwen3-wikitext2-protocol-6109964.json"
test_tokens="$protocol_root/qbbnew-qwen3-wikitext2-tokens-6109964.safetensors"
```

## Stage 0: environment and input preflight

Keep GPTQModel in a quantization-only venv because its NumPy constraint differs
from the already frozen PPL environment. On the target image, which already
provides CUDA PyTorch 2.8, prepare it with the explicitly pinned
`torchao==0.16.0`. Do not allow pip to select torchao 0.18: its Python API uses
PyTorch interfaces absent from 2.8.

```bash
python3 -m venv --system-site-packages /opt/fluxbin-gptq-w3-venv
/opt/fluxbin-gptq-w3-venv/bin/python -m pip install \
  -r "$project_root/infra/runpod/requirements-gptq-w3-v1.in"
/opt/fluxbin-gptq-w3-venv/bin/python -m pip freeze \
  > "$gptq_root.environment-freeze.txt"
```

Before the long quantization, run the input-only check using the accepted C4
manifest and token file paths:

```bash
/opt/fluxbin-gptq-w3-venv/bin/python \
  "$project_root/scripts/run_qwen3_8b_w3_gptq.py" \
  --snapshot-root "$snapshot_root" \
  --calibration-manifest "$calibration_manifest" \
  --calibration-tokens "$calibration_tokens" \
  --output-dir "$gptq_root" \
  --validate-only
```

This checks model revision/file hashes, architecture, accepted calibration
hashes, 256 x 2048 shape and token tensor hash. The real run additionally
requires an A100 80GB PCIe or SXM4 (CUDA capability 8.0) and the frozen package
versions. The accepted BF16/QBB PPL reproduction gates remain mandatory on
either A100 variant.

## Stage 1: create and reload the W3 artifact

```bash
/opt/fluxbin-gptq-w3-venv/bin/python -u \
  "$project_root/scripts/run_qwen3_8b_w3_gptq.py" \
  --snapshot-root "$snapshot_root" \
  --calibration-manifest "$calibration_manifest" \
  --calibration-tokens "$calibration_tokens" \
  --output-dir "$gptq_root"
```

Acceptance gates before PPL:

- 252 and only 252 packed GPTQ modules;
- every target has `qweight`, `scales`, `qzeros` and `g_idx`;
- no quantized embedding, `lm_head` or unclassified Linear;
- post-save reload uses `GPTQ_TORCH`, not Triton;
- every reloaded module confirms that its optional Triton dequantizer is
  disabled;
- 36 decoded layer files, 252 BF16 finite tensors and exactly
  6,945,767,424 decoded weights;
- raw and decoded file SHA-256 values are recorded in `manifest.json`.

GPTQModel 7.4.0 checkpoint/resume does not support dense Qwen3, so this process
must run in a durable session and cannot claim layer-resume safety. A stopped or
failed run is not accepted and must use a fresh output directory.

## Stage 2: derive the QBB FP16-scale artifact

```bash
/opt/fluxbin-venv/bin/python -u \
  "$project_root/scripts/convert_qwen3_8b_qbb_fp16_scales.py" \
  --source-acceptance "$qbb_root/acceptance.json" \
  --source-result "$qbb_root/result.json" \
  --source-payload-dir "$qbb_root/payloads" \
  --output-dir "$qbb_fp16_root"
```

Each layer is committed atomically. The runner verifies the accepted parent
hashes, the 36 payload hashes, all 756 fixed sign/index tensors across the
model, all 1008 FP16 scale tensors, and finite dense-BF16 reconstruction.

## Stage 3: run the four PPL arms

Use the original frozen PPL environment, not the GPTQ quantization venv:

Two execution policies are available:

- `formal-a100` is the default. It accepts only the frozen A100 80GB names and
  capability 8.0, and requires BF16 and current-QBB PPL to reproduce the saved
  A100 values within `1e-6`.
- `same-device-quality` accepts only RTX PRO 4500 Blackwell with capability
  12.0. It reruns all four arms on that one card and uses only the within-run
  PPL differences. The saved A100 BF16/QBB values are reported but not enforced;
  this mode is neither exact A100 reproduction nor latency evidence.

Both policies keep the frozen evaluator, artifacts and exact package versions.
The provider's `CUDA 13.0` label is not itself an accepted software environment:
the runner still requires Python 3.12, PyTorch `2.8.0+cu128`, Transformers
`5.14.1`, Datasets `5.0.0` and Safetensors `0.8.0`.

On a new Pod, first mount the retained volume at `/workspace` and inspect the
actual GPU, capability, driver and base PyTorch before installing anything:

```bash
nvidia-smi
python3 - <<'PY'
import platform
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

For the planned RTX PRO 4500 run, stop before creating the evaluator venv if
the reported name is not RTX PRO 4500 Blackwell, capability is not `(12, 0)`,
or base PyTorch is not exactly `2.8.0+cu128`. Do not silently accept a different
torch build supplied by a CUDA 13.0 template. Once these checks pass,
`/opt/fluxbin-venv` is recreated on container disk with only the pinned add-ons:

```bash
project_root=/workspace/repos/marlin-style-fluxbin
python3 -m venv --system-site-packages /opt/fluxbin-venv
/opt/fluxbin-venv/bin/python -m pip install \
  -r "$project_root/infra/runpod/requirements-linear-a100-v1.lock"
/opt/fluxbin-venv/bin/python -m pip install --no-deps -e "$project_root"
/opt/fluxbin-venv/bin/python -m pip check
```

Do not install GPTQModel into this evaluator environment. The raw W3 artifact
has already been decoded and validated. Before launch, require a clean checkout
at the intended revision and confirm that both the new job directory and formal
output directory are absent.

The prepared wrapper records the Git revision, environment, GPU, exact command,
PID, log, status and exit code, and refuses to overwrite existing output. Start
it inside tmux so the run does not depend on the SSH connection:

```bash
export FLUXBIN_PYTHON=/opt/fluxbin-venv/bin/python
export FLUXBIN_EXECUTION_POLICY=same-device-quality
tmux new-session -d -s qwen3-8b-w3-rate-distortion-v1 \
  "bash $project_root/scripts/run_qwen3_8b_w3_rate_distortion_job.sh \
  /workspace \
  /workspace/jobs/qwen3-8b-w3-rate-distortion-v1/ppl \
  /workspace/artifacts/qwen3_8b_w3_rate_distortion"
```

The direct runner command below is the frozen command embedded by that wrapper:

```bash
/opt/fluxbin-venv/bin/python -u \
  "$project_root/scripts/run_qwen3_8b_w3_rate_distortion_ppl.py" \
  --execution-policy "$FLUXBIN_EXECUTION_POLICY" \
  --snapshot-root "$snapshot_root" \
  --protocol-manifest "$protocol_manifest" \
  --token-artifact "$test_tokens" \
  --qbb-acceptance "$qbb_root/acceptance.json" \
  --qbb-result "$qbb_root/result.json" \
  --qbb-payload-dir "$qbb_root/payloads" \
  --qbb-fp16-result "$qbb_fp16_root/result.json" \
  --qbb-fp16-dir "$qbb_fp16_root" \
  --gptq-manifest "$gptq_root/manifest.json" \
  --gptq-dir "$gptq_root" \
  --output-dir "$result_root"
```

The scoring order is BF16, current QBB, GPTQ W3 g128, then QBB FP16 scales.
Every arm must report 146 blocks and 298,862 positions. A failed or partial
directory must not be relabelled as a completed comparison.

An RTX PRO 4500 completion is labelled
`completed_cross_device_pending_effect_size_review`. Acceptance uses the four
PPL values from that same run; old A100 anchors are diagnostic context only.

Bounded status checks:

```bash
cat /workspace/jobs/qwen3-8b-w3-rate-distortion-v1/ppl/status
tail -n 30 /workspace/jobs/qwen3-8b-w3-rate-distortion-v1/ppl/run.log
cat /workspace/jobs/qwen3-8b-w3-rate-distortion-v1/ppl/exit-code
```

Historical same-protocol measurements scored each arm in roughly 28 seconds,
but startup, full input hashing and four weight materializations add overhead.
Budget several minutes rather than treating 4 x 28 seconds as a guaranteed
wall-clock time.

## Output contract

The result root contains:

```text
summary.json
summary.md
bf16/{config,coverage,eval,storage,artifact_hashes}.json
qbb_current/{config,coverage,eval,storage,artifact_hashes}.json
gptq_w3_g128_sym/{config,coverage,eval,storage,artifact_hashes}.json
qbb_fp16_scales/{config,coverage,eval,storage,artifact_hashes}.json
```

`summary.md` contains the requested nominal bits, effective bits/weight,
serialized weight tensor storage, coverage, PPL, delta to BF16 and delta to
current QBB. The runner deliberately does not encode an arbitrary numerical
Case A/B/C threshold; the final decision is made from the observed effect size
after all provenance and coverage gates pass.

## Prepared-code validation

Local Apple Silicon validation does not establish CUDA or model-quality
correctness. It verifies the offline contracts only:

```text
121 tests passed; 8 CUDA-only tests skipped
```

The new tests cover the exact 8B weight count, QBB FP32/FP16 analytical rates,
GPTQ semantic 3.125-bit core, fixed-tensor preservation, packed GPTQ storage
categories, forced eager-Torch W3 decoding, BF16/QBB reproduction gates and
summary-table arithmetic.
