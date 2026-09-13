# Qwen3-8B accepted representative Linear results

2026-09-13. All five layer-0 representative targets passed independent review on
RunPod NVIDIA A100-SXM4-80GB. Source revision:
`310cd579fef73c64d193a3e60944729172a0da16`.
Runtime: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
Transformers 5.14.1 and Safetensors 0.8.0.

Each target uses the same original BF16 layer-0 inputs for its pure/hybrid
comparison. Each arm fits independently with the frozen two-base Hessian-OBQ
and hybrid-s8 contract. These are Linear reconstruction and calibration-error
results, not full-model PPL, packed-kernel timing or serving evidence.

| Target | [O,K] | Pure weight SSE | Hybrid weight SSE | Weight SSE reduction | Calibration output-loss reduction | Run seconds | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| o_proj | [4096, 4096] | 1621.227050 | 1491.531268 | 7.9999% | 20.8789% | 69.13 | 19.56 |
| q_proj | [4096, 4096] | 2130.508050 | 1964.954049 | 7.7706% | 49.4600% | 74.20 | 19.56 |
| k_proj | [1024, 4096] | 656.165264 | 599.533121 | 8.6308% | 52.6200% | 71.71 | 19.52 |
| gate_proj | [12288, 4096] | 6161.776486 | 5433.972562 | 11.8116% | 26.4970% | 61.16 | 19.69 |
| down_proj | [4096, 12288] | 6859.020478 | 5579.336292 | 18.6570% | 31.6725% | 48.51 | 20.58 |

Run time includes snapshot hash checks, model loading, calibration capture,
quantization and artifact verification; it is not isolated kernel latency.

## Acceptance

Every target has finite metrics, strict hybrid improvement in both weight SSE
and same-Hessian calibration output loss, exactly zero refinement outside the
selected columns, sorted valid group-local indices, and 10 verified payload
tensors. Independent review rehashed the result/config/source/payload records,
loaded and decoded the stored payload on CPU, and recomputed weight SSE against
the hash-pinned original BF16 target (agreement tolerance 1e-6 relative/absolute).
Calibration output loss was checked from the finite runner metrics; independent
review did not recapture activations or recompute that Hessian-based metric.
Producer status remains `completed_pending_review`; separate acceptance JSONs
record the completed review.

C4: 256 x 2048, seed 20260902, one pinned training shard (356,317 documents).
The separately generated 8B token artifact happened to match the historical
32B token-file hash; it was not substituted by renaming a 32B manifest.

- Calibration manifest SHA-256: `fe554afbd625b7746b0295d3a269825fca869bb7a7c159aaf469f1b122f387f9`.
- Calibration token-file SHA-256: `a31bfd489dccd7f4ea2cdc04e245f4d47229ca0fe788c4690d061a17fe47e2e2`.

## Artifact identities

- `o_proj` result: `d4b659516864e48e1b32d68a57a6c634cd4c02a32f8c28a8046c5e3db198e0d8`; payload: `589a525097a99ec3b03fe5d0d2a03d17fba247a8b930246b2fc5cc3f84658e29`.
- `q_proj` result: `90684b00947d9dbb2c220bcca57c0cbb306b7b08171558912b8505f558d60729`; payload: `390de74a4e08928fea7e4868156265f54140fc23ee866ca6e22cc9b58d1dcc6e`.
- `k_proj` result: `9c17c638e3405e8e134902af51c7c8fefda6bde7cc37670bf3d7cd2016a184fa`; payload: `6ccb2f91cd28cdc34651f719cc7fced70d5fd64cd36490286d091f113c08ef24`.
- `gate_proj` result: `ad499baf6901f13740c86bd56911b8d26e94aacb8e0ad5fb1dd2631c7c32f5e2`; payload: `68c4c3dbfa6875b66cf98a698b34ad6f2062033a6f78e9f6eb64d4d23451848a`.
- `down_proj` result: `c7dd558ed2df7f981308dfb101a9a84ec8bb0025696bb5898eaf11e2437451a6`; payload: `59b49966d155e4546588165d2575ead4609076312178319c353213e1f744c367`.

Server files are under the checkout's `results/qwen3-8b-linear-v1/` and
`artifacts/qwen3-8b-linear-v1/`. A private local metadata archive and audit script
are retained under `server_results/runpod_qwen3_8b_2026-09-13/`; weights were not
copied back to the Mac.

Full-model pure/hybrid quantization (36 layers and 252 Linears per arm) has
subsequently passed artifact integrity/reconstruction review with notes; see
[full-model results](QWEN3_8B_FULL_RESULTS.md). PPL and deployment remain gated.
