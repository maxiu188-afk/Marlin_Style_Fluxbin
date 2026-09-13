# Qwen3-8B full-model artifact acceptance

Review status: **accepted_with_notes** (artifact integrity and weight reconstruction).

2026-09-13, RunPod NVIDIA A100-SXM4-80GB. Both pure and hybrid-s8 completed
36 layers / 252 transformer-block Linears / 6,945,767,424 weight elements per
arm. This accepts the quantization artifacts and weight reconstruction only.
Matched BF16/pure/hybrid WikiText-2 PPL remains pending; no packed CUDA,
end-to-end inference, or serving performance was measured.

## Results

| Metric | Pure | Hybrid-s8 |
| --- | ---: | ---: |
| Weight SSE (stored payload reconstructed to BF16) | 727921.543228 | 637440.223930 |
| Relative Frobenius error | 0.3847081174 | 0.3600052186 |
| Branch-specific calibration output SSE | 113419.242014 | 102007.749789 |
| Quantization elapsed minutes | 22.07 | 30.40 |
| Peak allocated GPU memory (GiB) | 11.603 | 11.603 |
| Serialized payload bytes | 2181249504 | 2724825888 |
| Serialized payload GiB | 2.03145 | 2.53769 |
| Tensor-data bits per included weight (FP32 scales, no headers) | 2.51222826 | 3.13818359 |
| Serialized bits per included weight | 2.51232 | 3.13840 |

Hybrid reduces aggregate weight SSE by **12.4301%**. Improvement occurs in
249/252 Linears. The three regressions (zero-based layer indices) are:

| Layer / module | Pure weight SSE | Hybrid weight SSE | Increase |
| --- | ---: | ---: | ---: |
| 10 / self_attn.o_proj | 1528.885639 | 1550.189874 | 1.39345% |
| 18 / self_attn.o_proj | 1412.183232 | 1418.432125 | 0.44250% |
| 22 / self_attn.o_proj | 1443.889596 | 1471.201991 | 1.89158% |

Artifact integrity acceptance does not assert universal per-Linear improvement.
The frozen full-model config requires complete inventories, finite metrics and
payload round trips; per-tensor comparisons above are retained for review. The payload storage totals
include packed signs, FP32 scales, hybrid indices and safetensors headers. They
cover the selected block Linear weights, not embeddings, lm_head, a complete
standalone checkpoint, or inference memory.

Each arm independently propagates its own quantized layer outputs through the
same C4 token sequences. Later-layer inputs and Hessians therefore differ by
arm. The calibration output SSE totals are branch-specific diagnostics and
must not be interpreted as a matched-input comparison or a PPL improvement.
Full-model reconstruction uses BF16 materialization; the earlier representative
Linear weight metrics used FP32 reconstruction, so those values need not match.
Run duration includes setup, calibration capture, fitting and artifact I/O;
it is not kernel latency. Peak allocated memory is PyTorch's reported peak,
not total device usage.

## Frozen inputs and provenance

- Source revision: `d8a2eff231dee7f0f4c4822179685672d60e10f9` (clean remote checkout).
- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Resolved config SHA-256: `596c85f9e5e5427598eb039c9a5a5210860cc421f640ce6c7c70d93a51066d7f`.
- Calibration: pinned C4 training shard, seed 20260902, 256 x 2048 tokens.
- Calibration manifest SHA-256: `fe554afbd625b7746b0295d3a269825fca869bb7a7c159aaf469f1b122f387f9`.
- Calibration tokens SHA-256: `a31bfd489dccd7f4ea2cdc04e245f4d47229ca0fe788c4690d061a17fe47e2e2`.
- Pure result SHA-256: `3ab11878c60616523f122f7d11816378c29618a2c75b4f863823543c24e2702f`.
- Hybrid result SHA-256: `e484b9350c68827f129d1724264a08d54c968286793ec45063964a60d821d65b`.

The five accepted Linear gates bind the full config. Both full arms fitted all
36 layers freshly (zero resumed layers), with group size 128, frozen two-base
Hessian-OBQ, 50 ALS iterations, and eight selected columns per group for hybrid.
Runtime: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Transformers 5.14.1.
The environment reuses the RunPod template and resides on container disk.

## Acceptance method and limitations

A separate read-only CPU audit rehashed source/config, all 72 payloads and 72
layer metadata files, and every stored tensor; verified complete inventories,
finite values, shapes, dtypes, and valid sorted unique group-local indices;
then loaded the pinned original BF16 targets and reconstructed all 504 stored
Linear payloads with the versioned decoder. Recomputed weight SSE and target
norms agreed with producer records within 1e-6 relative/absolute tolerance;
maximum observed relative SSE discrepancy was 3.2544e-16. The audit took
430.83 seconds. Tensor-data bpw above is the logical content of the current
experimental representation, not a proposed deployment layout.
All layer aggregates must reproduce the final result summaries. This replays
stored artifacts independently of fitting; it is not a second decoder
implementation. Hessian captures and calibration output losses were inspected
for structure/provenance/finiteness but not independently replayed.

Audit startup exposed an operational script-name collision: job-local
`queue.py` shadowed Python's standard library when the audit was invoked by
filename in that directory. Its import attempted a duplicate queue invocation;
the runner refused to overwrite the existing pure result before quantization.
This replaced the pure log, PID, exit code and queue status with failed-retry
records. Original result JSONs, source manifests and payloads were retained and
verified. Thus the current pure exit code does not describe the original run,
and its original execution log is unavailable. The final audit uses `runpy`
from the repository root to avoid adding the job directory to the import path.
The failed attempts and their provenance are retained rather than relabelled.

After completing all integrity and reconstruction checks, the first audit
reported failure on an additional reviewer assertion that all 252 hybrid
Linears must improve. That assertion is absent from the frozen full config's
`decision` contract. The final review retains the failed report, the three
regressions, and the audit source hash; it accepts integrity/reconstruction
with notes and explicitly does not accept universal improvement or model quality.

Producer results retain `completed_pending_review`; the separate `acceptance.json`
is the authoritative artifact-review outcome. Private metadata, manifests,
logs and audit source are archived under
`server_results/runpod_qwen3_8b_2026-09-13/`; no weights were downloaded to the Mac.
Remote results live in `results/qwen3-8b-full-hessian-obq-s8-v1/` and payloads in
`artifacts/qwen3-8b-full-hessian-obq-s8-v1/`, relative to the remote checkout.

Subsequent update: the matched 8B BF16/pure/hybrid WikiText-2 evaluation has
been ported, tested and submitted separately; see `QWEN3_8B_PPL_GUIDE.md`.
Results remain pending review. Deployment requires the frozen quality gate
(relative PPL gap to matched BF16 <= 5%; > 10% blocks deployment).

- Audit source SHA-256: `126e21fab6d6f42acb53e532d3b16e437fcbe3bfe851963e50bfaaf9a4339cf4`.
- Portable manifest SHA-256: `190bf13e31b0b2e3c35fce02f1c69f1b36c577b365e6f07353d5d5c0905b5854`.
- Acceptance record SHA-256: `7a8a858f5fc33bbc42eebe3db1b51ff14a696e1aa0b788385d2f8bf218254ea8`.
