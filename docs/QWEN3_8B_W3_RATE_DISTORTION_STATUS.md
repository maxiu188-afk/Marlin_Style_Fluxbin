# Qwen3-8B W3/QBB rate--distortion artifact status

Updated: 2026-09-16. This records artifacts that are safe to retain on network
volume `34au39ljvf` before the A100 PCIe compute instance is closed. It is not a
PPL result and does not yet decide between uniform W3 and binary-base QBB.

## Current gate status

| Gate | Status | Evidence |
|---|---|---|
| QBB FP16-scale conversion | passed | 36 layers, 252 Linears, fixed sign/index hashes unchanged |
| GPTQ quantization coverage | passed | 36 layers, 252 Linears, 6,945,767,424 weights |
| GPTQ standard checkpoint reload | passed | GPTQModel 7.4.0 selected `TorchLinear=252` |
| GPTQ dense BF16 decode | passed | 36 layer files, 252 finite BF16 tensors, post-write tensor hashes checked |
| Frozen four-arm input/hash preflight | passed | `FLUXBIN_W3_RATE_DISTORTION_PREFLIGHT=passed` |
| Four-arm WikiText-2 PPL | **not run** | `/workspace/artifacts/qwen3_8b_w3_rate_distortion/` not created |

The default formal A100 PPL run must still reproduce BF16
`9.724944980689296` and current QBB `13.169788494766266` within `1e-6`, then
score GPTQ W3 and QBB FP16 scales over 146 blocks and 298,862 positions.

Because the next available GPU is planned to be an RTX PRO 4500 Blackwell, the
runner also has an explicit `same-device-quality` policy. That policy requires
the RTX PRO 4500 name and capability 12.0, reruns all four arms, and treats the
old A100 values as report-only anchors. Its result can support only within-run
quality/rate comparisons; it is not an A100 reproduction or speed result. The
frozen package versions remain mandatory, and no PPL result exists yet.

## GPTQ storage

The semantic quantized tensor storage is measured from the recovered standard
checkpoint, not inferred from nominal 3-bit packing:

| Tensor category | Bytes |
|---|---:|
| packed `qweight` | 2,604,662,784 |
| FP16 `scales` | 108,527,616 |
| stored `qzeros` | 20,348,928 |
| `g_idx` | 5,308,416 |
| **total** | **2,738,847,744** |

For 6,945,767,424 target weights this is **3.154551630 bit/weight**. The five
whole-model safetensors files occupy 5,228,911,856 bytes because they also
contain unchanged embeddings, `lm_head`, normalization weights and file
headers. The 36 decoded BF16 layer files occupy 13,891,559,904 bytes and are
evaluation material, not low-bit storage.

The completed QBB FP16-scale tensor storage is 2,284,886,016 bytes, or
2.631687330 bit/weight. No quality comparison is valid until Stage 3 PPL runs.

## Save failure and recovery boundary

The original GPTQ job completed all 252 quantization rows, then exited 1 during
save. GPTQModel 7.4.0 sets `SUPPORTED_SPLIT_BY = {None}` while retaining a stale
error message that advertises `split_by='layer'`. Repository commit `63a4bc8`
omits this unsupported argument for future runs.

The same commit adds a recovery-only runner. It did not quantize again. It:

1. validated the pinned source and exact 256 x 2048 C4 token artifact;
2. required all 252 offloaded modules and all four packed fields;
3. replaced only the corresponding 252 dense source weights while retaining
   pinned pass-through tensors;
4. wrote a standard five-shard GPTQ checkpoint;
5. reloaded it through GPTQModel with the eager Torch dequantizer; and
6. decoded and re-read all 36 BF16 layer files before writing `manifest.json`.

The original offload and failed save stub remain retained for audit. This
artifact has status `completed_pending_review`, not an accepted quality result.

## Persistent paths and hashes

- GPTQ root:
  `/workspace/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1/`
- GPTQ manifest SHA-256:
  `96b59535af55f30fe4b3992baa565800bfaaf659975d3e5fb501af0bfc17e482`
- recovery job:
  `/workspace/jobs/qwen3-8b-w3-rate-distortion-v1/recovery/`, exit code `0`
- recovery log SHA-256:
  `a7d4649c77eafdda3934ab0a142144427936805a39efbc7ca744157aacffa2b5`
- recovery source revision:
  `63a4bc8b8a02e28fcd86be54a0b787c9941bb661`
- QBB FP16-scale root:
  `/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-fp16-scales-v1/`
- QBB FP16 result SHA-256:
  `03a3e52e902b6467c61559e0051b6da477e53c298e4aefb47f1715fe80502dff`

At the final observation, no tmux session or experiment process remained. The
compute instance can be closed while retaining the network volume. On the next
instance, sync the repository, attach the same volume, run the Stage 3 command
from `QWEN3_8B_W3_RATE_DISTORTION_RUNBOOK.md`, and do not rerun Stages 1 or 2.
