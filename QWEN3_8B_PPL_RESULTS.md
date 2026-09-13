# Qwen3-8B matched WikiText-2 PPL acceptance

2026-09-13. **Execution accepted; quality gate failed for both quantized arms.**
The completed experiment is valid negative quality evidence. Neither pure nor
hybrid-s8 may advance to deployment under the frozen gate.

| Arm | Mean NLL | PPL | PPL / matched BF16 | Relative gap |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2.2746942320 | 9.7249449807 | 1.000000x | baseline |
| Pure | 7.0470567890 | 1149.4706247072 | 118.198162x | +11719.8162% |
| Hybrid-s8 | 2.7814310411 | 16.1421044428 | 1.659866x | +65.9866% |

Hybrid lowers PPL by 98.5957% versus pure, but its 65.9866% gap to matched BF16
is far above both the 5% admission threshold and the 10% explicit deployment
block. The numerical 5% ceiling is 10.2111922297; neither quantized arm meets it.
A reconstruction improvement or a large gain over a severely degraded pure
baseline does not establish acceptable model quality. The cause of pure's
large degradation has not been isolated by this acceptance review.

## Protocol and execution

- RunPod NVIDIA A100-SXM4-80GB; Python 3.12.3, PyTorch 2.8.0+cu128,
  Transformers 5.14.1, datasets 5.0.0, safetensors 0.8.0.
- Source revision: `e45462aa155aeeedf72f98080a03f74852bc9120`.
- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Reused exact frozen WikiText-2 tokens: 299,078 tokens, 146 independent
  non-overlapping complete 2048-token blocks; 298,862 transitions per arm.
- Same-run BF16 baseline, followed by pure and hybrid. Each quantized arm
  overwrites all 252 block Linears with its own decoded BF16 weights. Other
  parameters remain original BF16. No historical 32B baseline is substituted.
- SDPA, batch 1, cache disabled, TF32 disabled; BF16 logits and FP32 cross entropy
  in chunks of 128 scored tokens. No cross-block transition is scored.
- Producer elapsed: 180.13 seconds total, including preflight, loading,
  materialization and scoring. Scoring: BF16 28.20s, pure 27.86s, hybrid 27.95s.
  All three report 15.9913 GiB peak allocated memory during scoring.
  These dense fake-quant timings do not establish packed inference speedup.

## Acceptance checks

The review rehashed the pinned original model snapshot, accepted full-model
review record, both quantization result/source ledgers, all 72 layer payloads
and metadata, the PPL source files, config and frozen protocol/token files.
Token tensor and block-stream hashes were recomputed by the versioned protocol
loader. Layer coverage and the two arms' original target identities matched.
The run records validation of 756 pure / 1764 hybrid internal payload tensors
and complete materialization of 36 layers, 252 Linears and 6,945,767,424 weight
elements per quantized arm.

All metrics are finite, all three scored exactly 298,862 transitions, and no
nonfinite block was recorded. Review independently recomputed mean NLL from
recorded total NLL and count, PPL from exp(mean NLL), relative gaps and the
frozen quality decisions. Final results agree with per-arm progress records
and the intact PPL log; the job exited zero from the clean pinned revision.

This acceptance did not replay forward passes or retain/recompute per-token
logits. Internal payload tensor checks are recorded by materialization; current
whole-payload hashes were rechecked against the accepted immutable files.
The earlier full-quantization pure-log loss remains a documented provenance
limitation; this separate PPL job has its own intact log and source manifest.
Producer status stays `completed_pending_review`; a separate `acceptance.json`
records `accepted_execution_quality_failed` without changing the raw result.

## Artifact identities and next boundary

- Acceptance SHA-256: `da2ff44aa67b5e70b9f05f2905413cf4045bc8e4add2a6f535e9716a7230b9ba`.
- Result SHA-256: `99e5b288d60c5fa4d398ca5690ef93d7611bd96deaf615bf9fb38f1bdf0e1ab3`.
- Audit source SHA-256: `bcf9ebc960f7878c3d376b334114bedad9943fcc93a576abe5a6460e46ba5aef`.
- Config SHA-256: `92828caf81da5f76cab66287367e1020c4703b788d257f17cd38951f02a36b79`.
- Accepted full review SHA-256: `7a8a858f5fc33bbc42eebe3db1b51ff14a696e1aa0b788385d2f8bf218254ea8`.
- Protocol manifest SHA-256: `8b61cbeaba8809b94bc6b7568cb45ede0d941d046c812f57f3156163e42069ae`.
- Token artifact SHA-256: `252938697260d7f7241f26a05b9822c1a2fd5e9ae0168a87e0d5345d4a111ee0`.

Server result/acceptance: checkout `results/qwen3-8b-ppl-v1/`.
Job log/launch/exit records: `/workspace/jobs/qwen3-8b-ppl-v1/`.
Private local metadata and review archive:
`server_results/runpod_qwen3_8b_2026-09-13/`. No model weights were downloaded.

The next permitted planning stage is algorithm-quality diagnosis and an
explicitly scoped quality-improvement plan. No backend, distillation, new
quantization or additional evaluation was launched by this review. The earlier
artifact integrity/reconstruction acceptance remains valid; it did not promise
a passing PPL gate.
