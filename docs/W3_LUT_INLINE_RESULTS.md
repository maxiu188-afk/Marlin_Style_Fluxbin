# GPTQ W3 inline LUT：A100 PCIe 第一轮结果

更新：2026-09-16。本文只记录 Qwen3-8B、M=1、inline W3 LUT 的 Linear 级证据。
后续完整模型 trial 已完成，但因 packed-vs-decoded 数值门失败而未获 correctness
acceptance；详见[完整模型结果与误差归因](W3_LUT_FULL_MODEL_RESULTS.md)。

## 结论

第一轮 4 shape × 3 row tile 已在 NVIDIA A100 80GB PCIe 上完成。正式指标统一为
同一次运行中的 CUDA Graph total per call；candidate Graph 包含 inline LUT build、
main compute 和 `finish_m1`。12/12 cell 全部通过正确性、重复一致性、finite workspace
和计时稳定性门。

- q/o、gate/up、down 的最佳 inline W3 分别达到原始 BF16 的 **1.731x、2.678x、
  2.616x**。
- k/v 最佳点为 0.982x 原始 BF16，仍慢约 1.8%；不能宣称 W3 对所有真实 shape
  都加速。
- 相对冻结的 QBB `v5_p1024/gps1`，W3 在 q/o、k/v 更快，在 gate/up、down
  分别慢约 6.4%、4.8%。
- 这些结果证明当前 inline W3 有实际 M=1 加速价值，但也表明 row tile 与输出维度
  强相关，不能用单一 R 覆盖四类 shape。

## 12-cell 正式结果

`relative_to_v5 > 1` 表示 W3 更快；所有数值均为同轮中位数。

| shape | R | W3 Graph total (us) | vs original BF16 | vs decoded W3 BF16 | relative to v5 |
|---|---:|---:|---:|---:|---:|
| q/o `[4096,4096]` | **256** | **13.23840** | **1.73125x** | **1.74595x** | **1.16403x** |
| q/o `[4096,4096]` | 512 | 13.46592 | 1.70868x | 1.70764x | 1.14270x |
| q/o `[4096,4096]` | 1024 | 14.84736 | 1.55582x | 1.53655x | 1.03759x |
| k/v `[1024,4096]` | 256 | 8.55168 | 0.96715x | 0.95394x | 1.38535x |
| k/v `[1024,4096]` | **512** | **8.41824** | **0.98236x** | **0.96758x** | **1.40715x** |
| k/v `[1024,4096]` | 1024 | 10.23488 | 0.80922x | 0.79877x | 1.14632x |
| gate/up `[12288,4096]` | 256 | 27.33504 | 2.34155x | 2.34361x | 0.83717x |
| gate/up `[12288,4096]` | 512 | 25.01888 | 2.55729x | 2.55790x | 0.89591x |
| gate/up `[12288,4096]` | **1024** | **23.85984** | **2.67800x** | **2.68346x** | **0.94013x** |
| down `[4096,12288]` | 256 | 28.91392 | 2.32334x | 2.31592x | 0.84385x |
| down `[4096,12288]` | 512 | 27.01568 | 2.48662x | 2.47771x | 0.90240x |
| down `[4096,12288]` | **1024** | **25.66432** | **2.61616x** | **2.60942x** | **0.95380x** |

最佳稳定 inline candidate 固定为：q/o R256、k/v R512、gate/up R1024、down
R1024。后续 profiler 只允许覆盖这四点及同 shape 的 `v5_p1024/gps1`。

## 正确性和来源

- canonical `g_idx/desc_act`：36 层、252 个 Linear、252 个非恒等 permutation；
  每组严格 128 列，decoded zero 唯一值为 4。
- 离线 artifact：integer code round trip、decoded zero、loaded-BF16 scale cast 和
  retained decoded-BF16 全部精确通过。
- 12 个 candidate 对 structural reference 均为 `max_abs_error=0`、
  `normalized_rmse=0`；对 retained decoded BF16 的 NRMSE 为 0.00181--0.00206，
  低于冻结的 0.005 门槛。
- 所有 cell 都 `repeat_exact=true`、`workspace_finite=true`，四臂计时相对跨度均
  低于 10% 门槛。

正式源码 revision 为 `d433cfa00fdb986c2d1348029a224b05f69a428f`。

| artifact | SHA256 |
|---|---|
| 4×3 JSON | `521dedf538f6344c0b59eb7d82bcdcb72c930ef1e083e7b88c8655bc57df00cd` |
| W3 layout manifest | `f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8` |
| canonical semantics JSON | `81e82ea1a926c9e39166e45f878744b0f250cae2a6d045429a3ed6b39e2ece92` |
| CUDA preflight JSON | `bcdd7f6217f4bdbb6d4ee0d29ded6ed33856cb3c7849b0ae73e1761105d06e60` |

服务器结果位于 `/workspace/results/qwen3-8b-w3-lut-inline-v1/`；私有本地备份位于
`server_results/runpod_w3_lut_inline_a100_pcie_2026-09-16/`，不提交 GitHub。

## Profiler 状态与 prepare 门

当前实例包含 Nsight Compute 2025.1.1，但宿主驱动为
`RmProfilingAdminOnly: 1`，容器没有 `CAP_SYS_ADMIN`。最小计数器 smoke 明确失败为
`ERR_NVGPUCTRPERM`，退出码 1；日志 SHA256 为
`cb2a5d867a2756e0c87a26a735d446591b51f3446c813ac2b8bc04e9f1957827`。

因此尚无 achieved occupancy、eligible warps、DRAM throughput、shared bank conflict
或 source-counter 证据，不能判断 LUT construction 是否为主要瓶颈。`prepare`
诊断分支仍未授权、未实现。

随后使用四个最佳 row tile 完成了 36 层、252 Linear 的 full-sequence Graph trial：
性能约为原始 BF16 的 1.49x，但 decoded-W3 correctness gate 未通过。逐层与四 shape
算术分解已将原因定位为 kernel structural 算术与 dense BF16 权重物化语义不同，
而不是 packing/permutation 错误。当前优先级是决定数值语义和 oracle；在此之前
不自动实现 prepare，也不把 profiler 当作下一项正式 candidate。
