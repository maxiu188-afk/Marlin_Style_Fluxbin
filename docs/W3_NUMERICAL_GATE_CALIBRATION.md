# 完整模型 logits 数值门的标定：0.005 在本 harness 的噪声地板以下

更新：2026-09-17。本页不是新的 GPU 运行，而是对**已有结果 JSON 的重新读取**。
结论是：当前完整模型 `packed-vs-decoded` 的 0.005 logits NRMSE 门限低于本 harness
自身的放大地板，因此它无法区分「后端实现正确」与「后端实现错误」。本页给出标定
数据、放大链条和门限重做建议；它**不**声称 packed kernel 已经正确。

## 标定：一个零量化的对照组同样过不了这道门

`original_bf16` 臂是未量化、未 packed 的原始 BF16 模型。它的
`dynamic_static_check` 比较同一个模型在**动态注意力 mask** 与**预计算静态布尔
mask** 两条路径下的最终 logits。两条路径在数学上完全等价，差异只来自 kernel 选择
与归约顺序。

| 臂 | prompt | dyn-vs-static NRMSE | max_abs | 门限 | 结果 |
|---|---|---:|---:|---:|---|
| `original_bf16` | 0 | **0.01431** | 0.4375 | 0.005 | 未通过 |
| `original_bf16` | 1 | **0.01153** | 0.5 | 0.005 | 未通过 |
| `decoded_w3_bf16` | 0 | 0.01726 | 0.6406 | 0.005 | 未通过 |
| `decoded_w3_bf16` | 1 | 0.02858 | 1.0 | 0.005 | 未通过 |

**一个完全不含本项目任何 kernel 的 BF16 模型，在一次数学等价的改写下就已经落在
0.0115--0.0143。** 作为对照，被判失败的 packed-vs-decoded 值为：

| route | p0 | p1 |
|---|---:|---:|
| `structural`（09-16） | 0.01683 | 0.01457 |
| `fast_corrected`（09-17） | 0.02076 | 0.02200 |
| `observed_exact`（09-17） | 0.01973 | 0.01702 |

prompt 0 上，packed 的 0.01683 只是 BF16 自身地板 0.01431 的 **1.18 倍**；而门限
0.005 是该地板的 **0.35 倍**。

## 独立复现：QBB 线在不同 GPU、不同算法上给出同一地板

这不是 W3 专有现象。完全独立的 QBB `v5_p1024/gps1` + prepared v2.1 线（A100 SXM4、
不同量化方案、不同 kernel）记录的是 packed/decoded NRMSE 0.0125357 / 0.0129711、
动态/静态注意力 NRMSE 0.01167--0.01359（见[结果总览](RESULTS_OVERVIEW.md)）。
两条线、两块卡、两套算术，四个数字全部落在同一个 0.012--0.014 区间。

因此这个量级是 **harness 的性质**（32 步自回归 decode 终点的 logits 敏感度），
不是任何一个后端的性质。

## 放大链条

用 09-16 的逐层与逐 Linear 归因（`error-attribution-prompt0-step0-fc76b75.json`，
prompt 0）可以把放大过程拆开：

| 位置 | packed-vs-decoded NRMSE | 相对上一级 |
|---|---:|---:|
| 单个 Linear，同一输入（252 个） | 0.00059--0.00264 | — |
| step-0 decode logits（穿过 36 层） | **0.0085596** | 约 4x |
| step-32 最终 logits（门限比较点） | **0.01683** | 约 2x |

同文件中 prefill logits 为 `exact: true`、NRMSE 0.0 —— 但这只说明 packed route 是
decode-only、prefill 两臂走同一条 dense 路径，**不构成 kernel 正确性证据**。

两级放大（深度约 4x、自回归约 2x）合计把 0.002 量级的局部差异推到 0.017。任何
非逐位一致的实现都会经历同样的放大，这正是零量化对照组也停在 0.0143 的原因。

## 这解释了 `±3` 修正为什么无效

`±3` BF16 舍入修正在 Linear 层面确实有效：RTX PRO 4500 上 down 从 0.001821 降到
0.000154（见[修正诊断手册](W3_LUT_BF16_SPLIT_RUNBOOK.md)）。但端到端指标由放大
后的残差主导，局部精度提升淹没其中，最终 NRMSE 反而从 0.01683 升到 0.02076 ——
这是噪声主导型指标的典型行为，不是修正写错了。修正路线的 252 条 route 覆盖、
`dense_fallback=0` 均已核验通过，排除了接线问题。

## 门限重做建议

现行门限不可能通过，继续按它迭代 kernel 只会产生误导性的负面结果。建议改为：

1. **相对门（推荐）**：要求 `packed-vs-decoded ≤ original_bf16 自身的
   dyn-vs-static`。自带同轮噪声标定，不需要人为选阈值，且随 harness 变化自动跟随。
2. **step-0 单步 logits 门**：去掉自回归放大一级。该值现在就能从已有归因 JSON 读出
   （0.0085596），只需把它提升为正式门并为对照组补测同一指标。
3. **逐 Linear 固定输入门**：已经存在且**已经通过** —— structural 对 structural
   reference 为 `max_abs_error=0`、`normalized_rmse=0`，12/12 与 46/46 cell 全过。
   这才是 kernel 正确性的直接证据，应当被提为主门。
4. **质量门**：packed 后端能否部署是 PPL / 下游任务的问题，不是 bit 门能回答的。
   现有 PPL 仍来自 dense BF16 解码路径，不能代替 packed 后端质量验证。

## 已实现的门（2026-09-17）

方案 1、2 的测量侧和计时门的鲁棒化已落地，未改动任何已发布数字。

**相对门**：`runner.relative_backend_gate` 以同轮同 prompt 的
`original_bf16.dynamic_static_check.logits.normalized_rmse` 为地板，逐比较写入
`packed_vs_decoded_relative_check`（含 `noise_floor_nrmse`、`ratio`、`limit`、
`passed`），并汇总为 `all_relative_backend_checks_passed` 与
`candidate_acceptance[*].relative_backend_checks_passed`。limit 由 protocol 的
`packed_vs_decoded_relative_limit` 声明，默认 **1.0**（严格读法：packed 的偏差不得
超过 harness 对自身的偏差）。**这个默认值没有按“让当前结果通过”来选** —— 见下表，
structural 在 1.18 / 1.26 处仍然不通过。

**逐步 NRMSE**：`acceleration_checks.stepwise_nrmse` 沿 `[1,1+steps,vocab]` 的 step 轴
逐位置计算，由 `compare_trace` 以 `stepwise_normalized_rmse` 报告（report-only，不参与
`passed`）。索引 0 是 prefill，1..32 是 decode step 0..31。因为 `dynamic_static_check`
本身也走 `compare_trace`，本页原先列为「唯一待补测量」的**对照组 step-0
dyn-vs-static 值，下一轮会自动产生**，方案 2 的阈值可直接由它标定。

**计时门鲁棒化**：`statistics_row` 增加 `trim_per_side`（protocol 的
`timing_trim_per_side`，默认 1），以去掉首尾各一个样本后的 `trimmed_relative_range`
作为 `stable` 判据；`median` 仍用全样本，所以**加速比数值不变**。原始
`relative_range` 与 `stability_rule` 一并保留，尖峰不会被藏起来。样本数不足以施加
声明的 trim 时回退到原始极差，且 `validate_protocol` 对此 fail-closed，不做静默降级。

### 对已归档结果的重放

用新门重算两轮已归档 JSON（不重跑 GPU）：

| 轮次 | route | prompt | 旧 stable | 新 stable | 加速比 | 相对门 ratio |
|---|---|---|---|---|---:|---:|
| 09-16 | `structural` | 0 | True | True | 1.4880x | 1.176 FAIL |
| 09-16 | `structural` | 1 | True | True | 1.4892x | 1.264 FAIL |
| 09-17 | `fast_corrected` | 0 | False | **True** | 1.3436x | 1.451 FAIL |
| 09-17 | `fast_corrected` | 1 | False | False | — | 1.908 FAIL |
| 09-17 | `observed_exact` | 0 | False | **True** | 1.2701x | 1.379 FAIL |
| 09-17 | `observed_exact` | 1 | False | **True** | 1.2698x | 1.476 FAIL |

09-16 两个 cell 的判定与加速比**完全不变**，即新门不追溯改写已发布结果。09-17 有
三个 cell 从 void 恢复为可发布的负面性能结果。仍然失败的
`fast_corrected` p1 是 wall-time 上有**两个**尖峰（390.1 与 434.0 ms，中位数 355.1），
trim 一个之后仍有 9.89% > 5%；同 cell 的 device timing 为 0.23%，稳定。这正是期望
行为：单次宿主干扰可以豁免，反复干扰不行。

相对门下所有 cell 仍然不通过，ratio 落在 1.18--1.91。这是一个**有意义的**负面结果：
它说明 packed 后端的偏差确实高于 harness 自身地板 18%--91%，而不是像旧门那样报告
一个相对不可达阈值的 3.4 倍。

## 边界

- 本页只重新读取已有 result JSON，未新增 GPU 运行，未改变任何已发布数字。
- 本页证明的是「0.005 门在 0.005 处没有分辨力」，**不**证明 packed kernel 正确。
  kernel 正确性的现有证据是 Linear 级 structural 逐位一致，与本页独立。
- dyn-vs-static 与 packed-vs-decoded 是两种不同的扰动，不能互相替代；它们的可比性
  仅在于**同一个终点、同一个指标、同一个门限**，因此可以用前者标定后者的可达性。

## 来源

两轮 result JSON 的 `decoded_w3_bf16` 臂 `checked_static_audit.logits_sha256` 在
09-16 与 09-17 之间逐位一致（p0 `2be3a438...`、p1 `3fc2c614...`），
`w3_layout_manifest_sha256` 相同，因此跨轮 NRMSE 对比可比。

| artifact | SHA256 |
|---|---|
| 09-16 full-model result JSON | `c68811aa065b34d497f3113e0bc873684624ee0c8fc8df199998a7e53856f423` |
| 09-17 corrected result JSON | `08271b47c7db3e5197557a6fef25af659cf90e885621e7d4660a99d3c3a2c0cd` |
| 09-16 逐层归因 JSON | `8620846807f7214912c752aeb79d6011f61892d71dd1c22a81e96ca96d87cbe6` |

本地私有备份：`server_results/runpod_w3_full_m1_a100_pcie_2026-09-16/` 与
`server_results/runpod_w3_corrected_full_m1_a100_pcie_2026-09-17/`，不提交 GitHub。
