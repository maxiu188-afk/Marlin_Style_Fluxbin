# Marlin-Style FluxBin current handoff

## Current objective

Build and validate an independent two-base rank-one binary weight algorithm,
retaining a pure two-base arm and adding a sparse `s=8` residual-refinement arm.
Only after algorithm-quality acceptance should the project implement a
Marlin-style packed CUDA backend and measure real deployment speed.

The historical v1 weight-only full-model/PPL evidence remains preserved, but
its quality result is negative. The active v2 path now aligns the major missing
algorithm settings from the FluxBin algorithms before any backend work.

## Active v2 calibrated algorithm contract

- C4 calibration with 256 sequences, as stated by the paper.
- Project-frozen 2048-token sequence length and seed `20260902`; the paper does
  not specify either detail.
- Accumulate the scalar-normalized equivalent of `H = 2 X^T X`, then use a
  damped Cholesky inverse. Damping is frozen at 1% of the mean Hessian diagonal,
  a documented project choice because the paper does not state it.
- Exactly two binary bases, group size 128, independent row/column scales, and
  exact four-pattern sign assignment remain unchanged.
- Pure two-base remains an independent full arm with no residual refinement.
- Hybrid-s8 selects 8 group-local columns before global decomposition using
  `sum_i(W_ij^2) / Hinv_jj^2`, with stable lower-index tie breaking.
- After quantizing each group, update every unprocessed column using that
  arm's group error and inverse-Hessian blocks.
- Pure and hybrid do not share a global payload: refinement changes the group
  error, therefore later propagated working weights and global blocks differ.
- The existing greedy initialization and 50-step ALS limits remain unchanged.
- No Shared-C, distillation, CUDA kernel, or serving integration.

## Historical v1 weight-only contract

- Exactly two binary bases in `{-1,+1}`.
- Group size 128 along the Linear input dimension.
- Each base has independent row and column rank-one scales.
- Exact joint search over four two-base sign patterns.
- FP32 optimization and saved algorithm scales.
- Retain the complete pure two-base arm as a separately reportable output.
- Hybrid-s8 selects 8 columns in every 128-column group by the pure arm's
  residual column SSE over output rows, using stable lower-index tie breaking.
- Store selected group-local indices in ascending order.
- Fit a second independent two-base rank-one decomposition only to the selected
  residual columns; non-selected columns must remain bit-exactly unchanged from
  the pure arm's reconstruction.
- No Shared-C, Hessian error propagation, calibration data, distillation, CUDA
  kernel, or serving integration.
- FluxBin is an algorithm-form source only. Its accuracy tables, calibration
  choices, kernel claims, and internally inconsistent statements are not
  project evidence or acceptance targets.

## Evidence gates

1. Local/static plus synthetic Hessian, saliency and OBQ algebra checks.
2. Pinned C4 256x2048 calibration token artifact with hashes and sample ledger.
3. One real Qwen3-32B Linear emitting independent pure two-base OBQ and
   Hessian-salient hybrid-s8 OBQ outputs on Isambard GH200.
4. Layer-sequential full-model quantization emitting both independent arms,
   manually authorized after gate 3 review.
5. Dense fake-quantized WikiText-2 PPL, manually authorized after gate 4.
6. Packed backend correctness and performance in separate later phases.

No runner automatically launches its successor.

## Active v2 implementation status

- Implementation revision `549394b470b5772851728f53960f91092d140518`
  is pushed to `origin/main` and was fast-forwarded into the clean Isambard
  main checkout before submission.
- `src/fluxbin_style/hessian_obq.py` implements normalized Hessian
  accumulation, relative damping plus Cholesky inversion, Hessian saliency,
  block OBQ propagation, and independent pure/hybrid groupwise quantizers.
- The original initializer and 50-step solver are reused without changes.
- Isambard CPU numerical tests pass 21/21, including six new Hessian/OBQ tests.
- A tiny Qwen3 decoder smoke test captured the expected o-projection Hessian
  through the actual Transformers 5.14.1 forward API.
- Calibration config:
  `configs/calibration/qwen3_32b_c4_256x2048_v1.json`.
- Single-Linear v2 config:
  `configs/experiments/qwen3_32b_single_linear_hessian_obq_s8_v2.json`.
- The target remains `model.layers.0.self_attn.o_proj.weight`, shape
  `[5120,8192]`. Both arms are written into the same result payload but have
  separate global signs/scales.
- C4 materialization job `6262354` failed `1:0` after 25 seconds because
  `datasets` compared the deliberately single-shard train input against C4's
  complete train+validation split metadata and raised `ExpectedMoreSplitsError`.
  All 21 preflight tests passed, and 356,317 rows from the pinned shard were
  parsed before the metadata-only failure. Its logs are
  `logs/qwen3-32b-c4-calibration/fluxbin-c4-q32-6262354.{out,err}` and its
  canonical artifact directory is
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/calibration/qwen3-32b-c4-calibration-256x2048-v1/`.
- Calibrated single-Linear job `6262355` was cancelled automatically without
  allocation or execution because its `afterok:6262354` dependency failed. Its
  logs are
  `logs/qwen3-32b-single-linear-hessian-obq-s8/fluxbin-obq-q32-s8-6262355.{out,err}`
  and its result directory is
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-hessian-obq-s8-v2/`.
- Revision `903744789a201c32efbbad0390df5858ab577fa4` fixes only the bounded
  single-shard loader by setting `verification_mode=no_checks`; the pinned C4
  revision, file, sampling, tokenizer, seed, sample count and sequence length
  are unchanged. The server checkout was clean and all 21 tests passed again.
- Retry calibration job `6271395` was submitted with a 30-minute limit; the
  job completed `0:0` in 36 seconds on `nid011157`. Retry single-Linear job
  `6271396`, dependent on `afterok:6271395`, completed `0:0` in 9m45s on
  `nid010005`.
- Both jobs passed all 21 tests. The independent artifact audit rehashed the
  calibration manifest/token file, result/source manifest, and every payload
  tensor; reconstructed both arms directly from packed signs and scales;
  reloaded and hash-checked the original checkpoint tensor; and reproduced the
  stored weight SSE values.
- Calibration artifact: 256 sequences x 2048 tokens, 524,288 activation rows,
  Hessian shape `[8192,8192]`, token tensor SHA-256
  `f504da4f8b7fa56aa0d7c8971556c52eaee1395619a7fc4b10c2b4c04bb28599`,
  token file SHA-256
  `a31bfd489dccd7f4ea2cdc04e245f4d47229ca0fe788c4690d061a17fe47e2e2`,
  and manifest SHA-256
  `a89c967136a9d7964cfad3dcfc956dee2f6cae360ccb5858d254333a4d352675`.
- Pure two-base OBQ weight SSE is `3088.664830435032`; its calibration output
  loss is `2.7373215457945603`.
- Hessian-salient hybrid-s8 weight SSE is `2819.435390792252`, an `8.7166933%`
  reduction; its calibration output loss is `1.8975661604505283`, a
  `30.6779957%` reduction versus pure.
- Pure and hybrid global packed signs differ, confirming branch-independent
  propagated targets. Hybrid indices are `[64,8]`, sorted, unique and in range;
  the maximum refinement delta outside selected columns is exactly `0.0`.
- Result SHA-256:
  `93a3ae5ab0cf8ec6484e6019c1a3f0edb8cc8defa27ef4d3b5b627e6f25407a1`.
- Payload SHA-256:
  `bcddf5b77fb679f6784129c20bcd54e734a40369cccf78fbff5bc370d369583f`.
- Source-manifest SHA-256:
  `25141a0899e8f5668bf3f2ebbad3925cfd037b9bc22f94070cdb86b35589256e`.
- Acceptance: calibration and calibrated single-Linear execution/artifacts are
  accepted. The result clears the bounded implementation gate because hybrid
  improves both matched metrics while preserving the pure arm and sparse
  support contract. It does not establish full-model quality.
- Full-model, PPL and backend stages remain unlaunched. The next step is a
  layer-sequential full-model implementation with independently propagated
  pure and hybrid arms, followed by matched dense fake-quant PPL.

## Prepared calibrated full-model v2 run

- Config: `configs/experiments/qwen3_32b_full_hessian_obq_s8_v2.json`.
- Runner: `scripts/run_qwen3_full_hessian_obq_s8.py`; Slurm entrypoint:
  `scripts/run_isambard_qwen3_32b_full_hessian_obq_s8.sbatch`.
- Pure and hybrid-s8 are separate jobs and artifact directories. Each starts
  from the pinned BF16 checkpoint and propagates its own quantized layer output
  into the next layer; global payloads are never shared across arms.
- Within each current layer, one pre-quantization pass captures four
  exact-input Hessians: q/k/v, o, gate/up, and down. Sharing is restricted to
  modules whose Linear input tensors are exactly the same.
- Each completed layer is committed as one atomic directory containing one
  seven-Linear payload and metadata. Resume validates config, implementation,
  model revision, payload file and every tensor hash, applies the stored BF16
  materialization, propagates calibration inputs, and continues at the first
  missing layer.
- Initial execution also materializes weights through the packed artifact path;
  a tiny-Qwen test requires exact equality between first-pass and resumed BF16
  weights. This prevents floating-expression-order drift at resume/PPL time.
- The full preflight now passes 23 tests, including Qwen3 input/Hessian capture,
  layer propagation, both arm payload formats, atomic layer write, and exact
  resume replay. The accepted calibration/single-Linear hashes are bound in the
  full config.
- Each arm requests one GH200 for 8 hours. This is intentionally resumable
  rather than requesting a larger monolithic window. No job automatically
  launches PPL or backend work.

## First real-Linear gate

- Model: `Qwen/Qwen3-32B`.
- Snapshot: `9216db5781bf21249d130ec9da846c4624c16137`.
- Tensor: `model.layers.0.self_attn.o_proj.weight`.
- Shape: `[5120,8192]`.
- Tensor SHA-256: `af8343c70597ac417bf2daa550282fdc9639a23b3496506df99d9f168fc6adec`.
- Comparison: greedy two-base row-scale initialization versus optimized
  independent rank-one row/column factors with joint sign reassignment.
- Required execution gates: pinned tensor/runtime, finite metrics, monotonic
  scale and assignment stages, payload round-trip hashes, and no automatic
  follow-up.

The retained 32B snapshot remains under the historical QBB-New artifact root
and must not be modified or duplicated:

```text
${PROJECTDIR}/${USER}/qbb-new/huggingface/hub/
  models--Qwen--Qwen3-32B/snapshots/
  9216db5781bf21249d130ec9da846c4624c16137/
```

## Environment boundary

- macOS/Apple Silicon: source review, static compilation, and small CPU tests
  when a local project environment exists.
- Isambard GH200: real weights, reconstruction, PPL, and future performance.
- Submit durable work through Slurm and perform one bounded startup check.
- Scheduler `COMPLETED` is not acceptance; review structured JSON, provenance,
  hashes, payload inventory, finite metrics, and the gate contract.

## Accepted pure two-base result

- Two-base reference implementation: revision
  `e94c751971e05c5dd1aa72723d8b59b8a59c2168`, pushed to `origin/main`.
- First real-Linear Slurm job `6249838` completed `0:0` on 2026-09-02 from that
  clean revision after all six synthetic tests passed.
- Greedy SSE: `2734.2239076773903`; optimized pure two-base SSE:
  `2325.262335431492` (14.9571354% reduction versus greedy).
- Optimized relative Frobenius error: `0.3381926`; cosine similarity:
  `0.9410769`; MSE: `5.54386e-5`.
- The solver reached the 50-iteration cap and was near a plateau, but did not
  meet its formal convergence threshold.
- Result SHA-256:
  `a8030dfcc7dc23a82825dc8d5b142211e63f2cd135e0c001e692dc0d52792ab8`.
- Payload SHA-256:
  `54d83c27214ba91a9b1ce3dbb60e234db19e741b803bb4c4414bec967732f37c`.
- Source-manifest SHA-256:
  `9f4d4678fd7b0f238c6cb3972a0549673807acacb683806494d3cf26b849cb6d`.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-v1/`.
- Logs:
  `logs/qwen3-32b-single-linear/fluxbin2-q32-linear-6249838.{out,err}` in the
  Isambard checkout.

## Submitted pure two-base 50-versus-200 convergence check

- The accepted job `6249838` exhausted all 50 iterations with stop reason
  `max_iters`; its final iteration still reported relative improvement
  `2.3623791e-6`, above the solver threshold `1e-6`.
- The diagnostic keeps the same target tensor/hash, seed, initializer,
  algorithm, grouping, FP32 scale math, convergence patience, tolerance, and
  assignment chunking. Only `max_iters` changes from 50 to 200.
- The accepted baseline JSON is bound by SHA-256
  `a8030dfcc7dc23a82825dc8d5b142211e63f2cd135e0c001e692dc0d52792ab8`.
- “50 steps was too small” means the 200-step final SSE is below the 50-step
  final SSE by more than the frozen `1e-6` relative comparison tolerance. A
  negative or very small effect remains a valid diagnostic result.
- Job `6261256` was submitted on 2026-09-02 from clean revision
  `3e641e5be1213fcb5fd8daea122d93b2617c9b39`. The server preflight passed all
  15 unit tests, Python compilation, Slurm syntax, and the baseline-result hash
  and iteration-history checks. The bounded startup snapshot observed
  `PENDING (Priority)` with 1 GH200 GPU, 8 CPUs, and a 15-minute limit.
- It uses the isolated server checkout
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/repo-single-linear-i200` so the
  already queued PPL job keeps its original source checkout unchanged.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-max-iters-200-v1/`.
- Logs:
  `logs/qwen3-32b-single-linear-i200/fluxbin2-q32-i200-6261256.{out,err}` in
  the isolated checkout.
- Acceptance and the answer to the 50-step question remain pending structured
  result review; the scheduler state alone is not evidence.
- No full-model, PPL, or backend follow-up is launched automatically.

## Prepared hybrid-s8 gate

- Parent: the hash-pinned accepted payload from job `6249838`; the global arm
  is loaded and independently reconstructed, not refitted.
- Target: the same Qwen3-32B `model.layers.0.self_attn.o_proj.weight` tensor.
- Payload retains global signs/scales and adds group-local `int16` indices plus
  refinement signs/scales.
- Required checks: accepted-parent hashes and inventory, target hash, finite
  metrics, strict SSE improvement versus pure two-base, unique/sorted indices,
  exact zero reconstruction delta outside selected columns, and payload
  round-trip hashes.
- The Slurm job runs all synthetic tests before loading the real tensor.
- Result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-single-linear-two-base-rank1-s8-v1/`.

## Accepted hybrid-s8 single-Linear result

- Job `6250990` completed the application path and all 9 tests; SlurmDBD did
  not expose its scheduler accounting record, so scheduler `State/ExitCode`
  remains independently unavailable.
- Pure global SSE: `2325.262335431492`; hybrid-s8 SSE:
  `2142.084815168282`, a `7.8777142%` reduction.
- Relative Frobenius error: `0.3381925903` -> `0.3245984495`.
- Selected columns contained `8.3860997%` of global residual energy and the
  refinement removed `93.9377595%` of their SSE.
- Result SHA-256:
  `c4eb7effe1ad89a05689f8abbc713d65392c78fea68a699d20d72bdb0de31390`.
- Payload SHA-256:
  `bc1e69b788d4cdccaf8c5f186fe32d87182ab4d8c7a7859b763bba2b6bf59fee`.

## Full-model reconstruction contract

- Scope: all 448 transformer-block Linears across 64 Qwen3-32B layers,
  totalling 31,205,621,760 weights. Embeddings, output head and non-matrix
  tensors are excluded by a fail-closed inventory.
- Each Linear computes both the pure global two-base arm and hybrid-s8 in one
  pass. The hybrid reuses that exact global payload and stores only refinement
  signs/scales/indices in addition.
- One metadata JSON and one payload per Linear provide bounded restart. Resume
  requires matching target, config, implementation and payload hashes.
- Both global and refinement signs are losslessly stored in
  `fluxbin-two-base-interleaved-2bit-v1`. This is an algorithm artifact format,
  not a CUDA backend format or performance claim.
- Expected packed payload is about 12,222,480,384 bytes (11.38 GiB), excluding
  small safetensors/JSON overhead. Aggregate algorithm cost is about 2.5078 bpw
  for pure global and 3.1334 bpw for hybrid-s8 with FP32 scales.
- Required gates include 448/448 inventory, exact single-Linear reproduction,
  strict hybrid SSE improvement for every tensor, zero delta outside selected
  columns, packed-sign round trips, finite/monotonic solver records, complete
  896-file artifact inventory, and aggregate metrics.
- PPL and backend jobs are not launched automatically.

## Current status

- Local static checks for hybrid-s8: Python compilation, JSON validation,
  Slurm shell syntax, and `git diff --check` passed. The Mac has no project
  PyTorch environment, so tensor tests remain an Isambard preflight gate.
- Hybrid-s8 real-Linear job `6250990`: application-level accepted as documented
  above.
- Hybrid-s8 logs:
  `logs/qwen3-32b-single-linear-s8/fluxbin2-s8-q32-6250990.{out,err}` in the
  Isambard checkout.
- Full-model reconstruction job `6253233` was submitted on 2026-09-02 from
  clean revision `5234079df1807cab8b9c8633070c9fe78a17c963`; the bounded
  startup check observed `RUNNING` on `nid010224` after config, GH200 runtime,
  and CUDA packed-sign round-trip preflight passed. It was entering the 14-test
  stage at the last observation.
- Its time limit was reduced from 16 hours to 12 hours after submission. The
  closest QBB-New 448-Linear Qwen3-32B reconstruction took
  `30897.34031198197` seconds (8h34m57s), so 12 hours keeps about 40% margin
  while avoiding an unnecessarily long resource request. The checked end time
  is `2026-09-03T01:46:28+00:00`.
- Full-model result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-full-two-base-rank1-s8-v1/`.
- Full-model resumable artifacts:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/artifacts/qwen3-32b-full-two-base-rank1-s8-v1/decompositions/`.
- Full-model logs:
  `logs/qwen3-32b-full-two-base-rank1-s8-v1/fluxbin-full-q32-s8-6253233.{out,err}`.
- Full-model application and independent artifact audit accepted: 448/448
  tensors strictly improved, 896/896 files and all packed decoded-sign hashes
  passed. Aggregate pure/hybrid SSE is
  `1907722.585722923/1762887.8386713415` (`7.5920235%` lower); relative
  Frobenius error is `0.3406372032/0.3274513676`.
- Full-model result SHA-256:
  `bf412e333e7ffe761c0627fecc6b2159ae19faa60f3dc48a125713fcd802084c`.
- Full-model source-manifest SHA-256:
  `6c418d7453565a784ecf3f2d83e9dd4a5cbe2abbb546e6de539d373538cac06a`.
- PPL job `6259037` completed all three arms on 2026-09-02 from clean revision
  `59fe6e7080eee720321c653fdc6102f4c6a5d5e1`. It reused the accepted QBB-New
  protocol of 146 non-overlapping 2048-token blocks and 298,862 scored
  transitions, and exactly reproduced BF16 reference PPL
  `7.6108390395557874` from job `6154681`.
- PPL materialization is dense BF16 fake quantization from the packed algorithm
  payload; it is not packed backend or speed evidence.
- Pure global two-base rank-one PPL is `147.53151938752475` (`19.3844x` BF16).
  Hybrid-s8 PPL is `24.215285155527635`: an `83.5864%` PPL reduction versus
  the pure arm, but still `3.18168x` BF16.
- The application path completed all 448/448 materializations per quantized
  arm and 146/146 scoring blocks per arm in `238.8073` seconds. All metrics are
  finite and each arm scored exactly 298,862 transitions.
- Independent audit accepted execution and provenance: 15 non-shard files were
  rehashed, all 17 model-shard hashes matched the independently accepted
  full-reconstruction manifest, BF16 reproduced with zero difference, and the
  stored NLL/PPL relationships were recomputed successfully.
- Result SHA-256:
  `1c379ef97c39959dab59bcf5108b2e95e32456e6e4251b801d5d6788bbb810bc`.
- Source-manifest SHA-256:
  `a4f5f2321283f32d94476fe0c821de1196467d1eef3ce07482046e16448851d1`.
- PPL result directory:
  `${PROJECTDIR}/${USER}/marlin-style-fluxbin/results/qwen3-32b-two-base-rank1-s8-ppl-v1/`.
- PPL logs:
  `logs/qwen3-32b-two-base-rank1-s8-ppl/fluxbin-ppl-q32-s8-6259037.{out,err}`.
- Acceptance boundary: the PPL execution evidence is accepted, but the
  algorithm-quality conclusion is negative. No numeric quality threshold was
  predeclared, yet hybrid-s8 PPL `24.22` versus BF16 `7.61` is not sufficient
  evidence to proceed to deployment optimization.
- Backend: not implemented and not authorized by the current quality result.
