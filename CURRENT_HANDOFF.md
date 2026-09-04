# Marlin-Style FluxBin current handoff

## Current status

The assignment-overhead repair, complete Qwen3-32B v3 full-model
reconstruction, exact replay gate, and matched WikiText-2 PPL execution are
accepted. Hybrid-s8 is materially better than pure two-base OBQ, but it remains
above BF16 quality; packed-backend and serving work are not authorized by this
result.

Current accepted checkpoints:

- full-model source revision:
  `412764e04cc5c997e7bc135ed52b128d08897a61`;
- PPL source revision: `e6921426e8e2249268f0861505556c12adc603e0`;
- model: `Qwen/Qwen3-32B` at
  `9216db5781bf21249d130ec9da846c4624c16137`;
- full-model config:
  `configs/experiments/qwen3_32b_full_hessian_obq_s8_v3.json`;
- PPL config:
  `configs/evaluation/qwen3_32b_wikitext2_full_hessian_obq_s8_v3.json`;
- artifact id:
  `qwen3-32b-full-hessian-obq-s8-v3-assignment-optimized`.

No backend or distillation job was launched.

## Frozen calibrated algorithm contract

- C4 calibration uses 256 sequences. Sequence length 2048 and seed `20260902`
  are explicit project choices because the paper does not specify them.
- Accumulate the scalar-normalized equivalent of `H = 2 X^T X`; apply 1%
  mean-diagonal damping before Cholesky inversion. The damping value is also a
  project-frozen choice.
- Use exactly two binary bases, group size 128, independent row/column scales,
  exact four-pattern sign assignment, the existing greedy initializer, and a
  50-step ALS limit.
- Pure two-base is a complete independent OBQ arm with no residual refinement.
- Hybrid-s8 chooses 8 group-local columns before global decomposition using
  `sum_i(W_ij^2) / Hinv_jj^2`, with stable lower-index tie breaking, and fits
  sparse residual refinement.
- After each group, each arm updates all unprocessed columns using its own group
  error and inverse-Hessian blocks.
- Each branch propagates its quantized hidden states into the next layer. Pure
  and hybrid therefore do not share global payloads or later-layer Hessians.
- No Shared-C, distillation, packed CUDA kernel, or serving integration.

## Accepted assignment-overhead repair

The old fixed 16-row assignment chunk caused about 10.24 million synchronizing
`.item()` calls per pure layer and 17.29 million per hybrid layer. Revision
`eef867dfa37ad2b5e2cd848eb4330bd99c46d313` introduced a 1 GiB adaptive
temporary-memory budget and accumulated assignment-change counts on device.
The active group-local OBQ fit now processes all 5120 output rows in one chunk,
while the legacy full-weight path remains memory-bounded.

Single-Linear regression job `6282732` completed `0:0` on Isambard GH200 in 51
seconds and reproduced accepted job `6271396` bit-for-bit:

- all 10 payload tensors and solver records matched;
- payload SHA-256 remained
  `bcddf5b77fb679f6784129c20bcd54e734a40369cccf78fbff5bc370d369583f`;
- application time: `571.1411166 -> 27.7813934` seconds (`20.5584x`);
- Slurm wall time: `00:09:45 -> 00:00:51` (`11.4706x`);
- peak allocated memory remained exactly `71,720,856,576` bytes.

This accepts the semantics-preserving assignment repair. It does not claim a
packed-kernel speedup.

## Complete v3 full-model execution

### Bounded replay and exact gate

The first v3 jobs regenerated exactly the previously accepted layer scope using
the repaired implementation:

| Job | Arm / scope | State | Slurm time | Application time |
| --- | --- | --- | ---: | ---: |
| `6283506` | pure, layers 0-10 | `COMPLETED 0:0` | `00:11:57` | `533.2232 s` |
| `6283507` | hybrid-s8, layers 0-5 | `COMPLETED 0:0` | `00:10:12` | `417.0263 s` |
| `6283508` | exact comparison gate | `COMPLETED 0:0` | `00:00:20` | - |

The repaired averages were `48.4748` seconds per pure layer and `69.5044`
seconds per hybrid layer. Against the matched pre-repair means of `2593.999`
and `4662.264` seconds per layer, this is approximately `53.51x` and `67.08x`.

Gate `6283508` passed both arms:

- pure layers 0-10: 11 layers, 231 payload tensors, 1,681,334,512 bytes;
  all payloads bit-exact and all algorithm metadata exact; comparison SHA-256
  `8297431e588dca751db37f224c723dd3d1cebd8576a97244239b2c51ba324f68`;
- hybrid layers 0-5: 6 layers, 294 payload tensors, 1,145,889,312 bytes;
  all payloads bit-exact and all algorithm metadata exact; comparison SHA-256
  `e6c8db53a459b896f431fb5b8baf3b55fd64da8e3575d6f314b1517aee58c686`;
- gate result SHA-256:
  `b3a97237f5352ecbdf786cf0f90230b4d470c3bc2c5c5357f3556e7ee723fe25`;
- `continuation_authorized=true`; neither PPL nor backend auto-launched.

Two earlier boundary-test jobs were stopped and quarantined after exposing an
inclusive layer-boundary bug. Revision
`412764e04cc5c997e7bc135ed52b128d08897a61` fixed the boundary before any
accepted v3 replay or continuation.

### Resume and final inventory

| Job | Arm | Replayed layers | New layers | State | Slurm time |
| --- | --- | ---: | ---: | --- | ---: |
| `6283509` | pure | 11 | 53 | `COMPLETED 0:0` | `00:45:14` |
| `6283510` | hybrid-s8 | 6 | 58 | `COMPLETED 0:0` | `01:06:45` |

Both final arms cover 64 layers, 448 Linears, and 31,205,621,760 weights.
Independent server-side acceptance rehashed all 128 metadata/payload pairs
(about 22.01 GB), validated every Safetensors inventory and internal tensor
hash, confirmed contiguous layers, and found finite metrics throughout.

| Metric | Pure | Hybrid-s8 | Hybrid change vs pure |
| --- | ---: | ---: | ---: |
| Weight SSE | `2,337,870.7642333917` | `2,117,499.284547236` | `-9.42616175%` |
| Relative Frobenius | `0.37708974135338075` | `0.35887739498900334` | `-4.82971144%` |
| Branch-calibrated output SSE | `1,155,273.3216937627` | `394,157.5088691871` | `-65.88188254%` |

The output-SSE comparison is branch-specific: later layers have different
propagated inputs and Hessians. It is not a same-Hessian comparison.

Final result SHA-256 values:

- pure: `4b9f5ab5c41bf7e266fc8a4c52072db9fb7869437bb871d3c55c14d243078e08`;
- hybrid-s8:
  `9a8753cdb99efc224651c5ca270efe050e44c58d4c1ccd66133c5d132924746a`.

The JSON status remains `completed_pending_review` because the producer is
fail-closed. Manual review is now complete and accepts the execution and
artifacts.

## Accepted WikiText-2 PPL gate

Job `6296175` ran on GH200 node `nid010773` from clean revision
`e6921426e8e2249268f0861505556c12adc603e0`; Slurm recorded `COMPLETED 0:0` in
`00:06:04`.

The run reused the accepted QBB-New WikiText-2 protocol without resampling:

- 146 non-overlapping 2048-token blocks;
- 299,078 tokens and 298,862 scored transitions;
- token SHA-256
  `c7a8c41e587561b93c8dd0b17224e6f20aa4270c9dab62151357cca88651ad9e`;
- BF16 reference job `6154681`, reproduced with absolute difference `0.0`.

| Arm | Mean NLL | PPL | Relative to BF16 |
| --- | ---: | ---: | ---: |
| BF16 | `2.0295734206653275` | `7.6108390395557874` | `1.000000x` |
| Pure | `2.876087261093014` | `17.744706766201215` | `2.33150467x` |
| Hybrid-s8 | `2.3450575398268865` | `10.433873073527254` | `1.37092284x` |

Hybrid-s8 reduces PPL by `7.3108336927` (`41.20008174%`) versus pure. Pure is
`133.150467%` above BF16 and is rejected as a final candidate. Hybrid-s8 is the
current best calibrated arm, but remains `37.092284%` above BF16; it is not
accepted as BF16-equivalent or production-ready.

All three arms scored exactly 298,862 transitions with no non-finite blocks.
For each quantized arm, materialization covered 64 layers, 448 Linears, and
31,205,621,760 parameters, and validated all payload and internal tensor hashes.
The application result elapsed time was `178.0604` seconds; the larger Slurm
time includes preflight tests, manifests, and hashing.

- PPL result SHA-256:
  `83b682aae91bbbde1b39b1b31c56716f2497501d220751b322ce6fb4dd861325`;
- PPL source-manifest SHA-256:
  `9644187b14a3e04462092bd6d5e9c2799d339975daa401ed2e91e77ff601e4eb`.

This is dense fake-quant PPL. It does not establish packed representation
correctness, kernel performance, end-to-end latency, throughput, or serving.

## Retention and private export

The obsolete partial full-model v2 artifact directory, result directory, and
jobs `6272553`/`6272554` logs were removed from Isambard. The corresponding
partial full-v2 data was also removed from local `server_results/`. The accepted
C4 calibration, WikiText-2 protocol/BF16 reference, and single-Linear oracle
remain because v3 provenance depends on them.

The private, Git-ignored `server_results/` bundle contains:

- v3 bounded-replay and final full-model result JSON/source manifests;
- the exact replay-gate JSON/source manifest;
- the v3 PPL result/source manifest;
- the associated full-model, gate, PPL, calibration, and single-Linear logs;
- calibration/protocol dependencies and transfer provenance.

The v3 full-model weights were deliberately not downloaded. Remote-to-local
checksum dry runs found no file-content differences. The current local
`provenance/SHA256SUMS` SHA-256 is
`26d72ab7d660b13d4334acd2ef4cbe47261180146595746e1db9199cb30fe70e`.
Nothing under `server_results/` belongs in Git.

## Implementation map

- `src/fluxbin_style/hessian_obq.py`: Hessian accumulation/inversion, saliency,
  OBQ propagation, and independent pure/hybrid groupwise quantization.
- `src/fluxbin_style/two_base_rank1.py`: two-base rank-one fit and adaptive
  assignment chunking.
- `src/fluxbin_style/qwen3_sequential.py`: Qwen3 layer capture, materialization,
  propagation, and resume support.
- `scripts/run_qwen3_full_hessian_obq_s8.py`: branch-specific full-model runner.
- `scripts/compare_full_hessian_obq_artifacts.py`: exact pre/post-repair gate.
- `scripts/run_qwen3_full_hessian_obq_s8_ppl.py`: v3 dense fake-quant PPL gate.
- `scripts/run_isambard_qwen3_32b_full_hessian_obq_s8.sbatch`: full-model Slurm
  entrypoint.
- `scripts/run_isambard_qwen3_32b_full_hessian_obq_s8_ppl.sbatch`: PPL Slurm
  entrypoint.

## Evidence boundary and next decision

The algorithm-quality ladder through dense fake-quant PPL is complete. The
accepted result says:

- the runtime repair preserved the algorithm exactly;
- both 64-layer branches are complete and internally valid;
- hybrid-s8 materially improves pure in reconstruction and PPL;
- hybrid-s8 still has a substantial gap to BF16.

There was no predeclared deployment-quality threshold, so the PPL execution is
accepted without claiming that the quality clears production use. Packed CUDA,
serving integration, and distillation remain unlaunched. Any next experiment
requires a separate explicit decision and acceptance contract.

## Environment boundary

- macOS/Apple Silicon: source review, static compilation, and small local tests.
- Isambard GH200: accepted real-weight, exact-gate, full-model, and PPL evidence.
- Isambard remains accessible through 2026-09-05; access is expected to end from
  2026-09-06.
- Scheduler `COMPLETED` alone is not acceptance. Structured JSON, provenance,
  hashes, inventory, finite metrics, and the declared gate must all pass.

## Historical note

The earlier v1 weight-only full-model result and PPL execution were valid but
negative: pure PPL `147.5315`, hybrid-s8 PPL `24.2153`, versus BF16 `7.61084`.
They are historical algorithm evidence only. The obsolete partial calibrated
full-model v2 run was superseded by the exact-gated complete v3 run and is no
longer retained.
