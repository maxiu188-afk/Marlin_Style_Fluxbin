# Qwen3-8B W3/QBB rate--distortion artifact status

Updated: 2026-09-16. The same-device four-arm quality comparison completed on
an RTX PRO 4500 Blackwell and selects **Case A**: at essentially the same stored
rate, uniform symmetric GPTQ W3 g128 is materially better than current QBB.
The evidence and model artifacts remain on network volume `34au39ljvf`.

## Current gate status

| Gate | Status | Evidence |
|---|---|---|
| QBB FP16-scale conversion | passed | 36 layers, 252 Linears, fixed sign/index hashes unchanged |
| GPTQ quantization coverage | passed | 36 layers, 252 Linears, 6,945,767,424 weights |
| GPTQ standard checkpoint reload | passed | GPTQModel 7.4.0 selected `TorchLinear=252` |
| GPTQ dense BF16 decode | passed | 36 layer files, 252 finite BF16 tensors, post-write tensor hashes checked |
| Frozen four-arm input/hash preflight | passed | `FLUXBIN_W3_RATE_DISTORTION_PREFLIGHT=passed` |
| Four-arm WikiText-2 PPL | passed | 146 blocks and 298,862 positions for every arm; exit code 0 |
| Result/provenance hashes | passed | recorded and recomputed summary hashes agree |
| Shutdown readiness | passed | no tmux session, GPU process or experiment process remains |

## Four-arm result and decision

| Arm | Effective bits/weight | Serialized weight bytes | PPL | Delta PPL vs current QBB |
|---|---:|---:|---:|---:|
| BF16 | 16.000000 | 13,891,534,848 | 9.726488173 | -3.441421766 |
| current QBB | 3.138184 | 2,724,636,672 | 13.167909939 | 0 |
| GPTQ W3A16 g128 symmetric | 3.154552 | 2,738,847,744 | **11.266114820** | **-1.901795120** |
| QBB FP16 scales | 2.631687 | 2,284,886,016 | 13.168951166 | +0.001041227 |

GPTQ uses only 0.5216% more serialized weight storage than current QBB while
reducing PPL by 1.9018, or 14.4426%. This is the frozen protocol's Case A: the
current approximately 3.14-bit QBB point is practically dominated, so further
deep kernel optimization for this QBB format stops and subsequent backend work
should prioritize uniform 3-bit. QBB FP16 scales reduce storage by 16.1398%
with negligible PPL change, preserving a lower-rate trade-off point but not the
quality advantage needed to overturn Case A.

This is a same-device quality comparison, not an A100 reproduction or latency
result. The RTX run used `same-device-quality` on CC 12.0; the saved A100 BF16
and QBB anchors differed by 0.001543 and 0.001879 PPL and were report-only.
Those shifts are tiny relative to the 1.9018-PPL within-run effect. The formal
`formal-a100` path remains available for an optional exact-device replay.

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
2.631687330 bit/weight. Its measured PPL is 13.168951166.

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
- four-arm result root:
  `/workspace/artifacts/qwen3_8b_w3_rate_distortion/`
- PPL job:
  `/workspace/jobs/qwen3-8b-w3-rate-distortion-v1/ppl/`, exit code `0`
- PPL source revision:
  `f979f6b3f64aa7ccabd2d96cc5e953a83e2d7080`
- summary JSON SHA-256:
  `de5e12fcd7c1db5da1952717dd15a01c9c230b34a3d2b254e64d0fc05e88e8cd`
- summary Markdown SHA-256:
  `a3180d4c30a866cc2a0cb4a0dd40b07963801a910fa33bc455312bd7059f912f`
- persistent closeout archive:
  `/workspace/jobs/qwen3-8b-w3-rate-distortion-v1/closeout-rtx4500-20260916/`
- evidence archive SHA-256:
  `812ba530f9fb4a41f97776c952c5eb5ce4f5dfef8e34fb776729a03abe49bfe7`
- private local backup:
  `server_results/runpod_w3_rate_distortion_rtx4500_2026-09-16/`

At the final observation, no tmux session, GPU process or experiment process
remained. The RTX compute instance can be closed while retaining the network
volume. Do not rerun Stages 1--3 unless an explicit replication is requested.
