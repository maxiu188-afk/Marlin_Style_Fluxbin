# 当前结果总览

2026-09-13：Qwen3-8B 已完成 Linear、全模型量化、补偿修复、scale-only 蒸馏和
固定 step400 的 test PPL 验收。精度实验按用户决定暂停，后续转向加速研究。
用户已确认计算服务器关闭。当前尚无 packed CUDA、block 或全模型加速结果。

## Qwen3-8B test PPL

模型为 `Qwen/Qwen3-8B`，固定 revision
`b968826d9c46dd6066d109eabc6255188de91218`。量化范围为 36 层、252 个 Linear、
6,945,767,424 个权重；embedding、norm、lm_head 保留原模型参数。

| 权重版本 | WT2 test PPL | 结果记录 |
| --- | ---: | --- |
| BF16 | 9.724944981 | 各轮精确复现，本轮同测 |
| 原 pure | 1149.470624707 | [原始 PPL 验收](QWEN3_8B_PPL_RESULTS.md) |
| 原 hybrid | 16.142104443 | [原始 PPL 验收](QWEN3_8B_PPL_RESULTS.md) |
| 修复补偿 hybrid，未蒸馏 | 14.951611048 | [修复版 PPL 验收](QWEN3_8B_CONDITIONED_PPL_RESULTS.md) |
| 修复补偿 hybrid，蒸馏 step400 | **13.169788495** | [最终 test 验收](QWEN3_8B_DISTILLED_TEST_RESULTS.md) |

各轮使用相同冻结 token/block 协议：146 个非重叠 2048-token 块，298862 个预测位置。
表格汇总多个运行；最终一轮只同测 BF16 和 step400。原始 pure/hybrid 在 A100 SXM4，
修复及蒸馏版本在 A100 PCIe。旧版本没有在最终一轮重测。

补偿修复相对原 hybrid 降低 PPL **7.3751%**；蒸馏相对直接父版本再降低
**11.9173%**，两项合计相对原 hybrid 降低 **18.4134%**。
这支持混合表示和蒸馏改善本配置下 PPL 的阶段结论。最终值仍比 BF16 高
**35.4228%**，原质量门槛未通过；用户已授权研究性 correctness/benchmark 先行。

## 蒸馏与 validation（与 test 分开）

参考 QBB-New：仅训练 1008 个 scale 张量，固定符号、选列索引和其他参数。
Loss 为硬 next-token CE / 初始训练 CE，加上 36 个 block 的平均 feature MSE /
初始训练 feature MSE。logit MSE 用于候选筛选，训练 token loss 不是 KL。
400 条候选筛选 200 条训练，100 条独立合成验证；2 epochs、400 步、batch 1。

| Validation 指标 | step 0 | step 400 |
| --- | ---: | ---: |
| WT2 validation PPL | 15.272183 | 13.670310 |
| WT2 validation feature MSE | 27.899045 | 20.895968 |
| 合成 validation 归一化总 loss | 1.621944 | 0.630516 |

WT2 validation 使用 128 个 2048-token 块、262016 个预测位置，PPL 改善 **10.49%**。
固定保留最终 step400；step300 的 validation PPL 13.664953 略低，但没有保存，
也未据此选点。训练期间未评估 test，最终 test 是后续独立任务。
训练脚本记录模型装载/替换后的耗时 66.20 分钟、峰值 allocated 26.19 GiB；
这些数值不是推理加速测量。详见 [蒸馏验收](QWEN3_8B_DISTILLATION_RESULTS.md)。

## 验收与来源

| 阶段 | 已完成范围 | 证据 |
| --- | --- | --- |
| 代表性 Linear | 5 个目标通过 | [Linear 结果](QWEN3_8B_LINEAR_RESULTS.md) |
| 原始全模型 | pure/hybrid 各 36 层、252 Linear；完整性与重建 accepted_with_notes | [全模型结果](QWEN3_8B_FULL_RESULTS.md) |
| 修复补偿 | 3 个 Linear 诊断；6 项指标一致性通过 | [诊断记录](QWEN3_8B_HYBRID_PROBE_GUIDE.md) |
| 修复版全模型 | 252 个重建通过，最大相对 SSE 差异 3.2879e-16 | [修复全模型](QWEN3_8B_CONDITIONED_FULL_RESULTS.md) |
| 蒸馏 | 400 步、36 层导出；固定张量一致、1008 个 scales 更新 | [训练验收](QWEN3_8B_DISTILLATION_RESULTS.md) |
| 最终 test | 执行验收通过，原数值质量门槛未通过 | [test 验收](QWEN3_8B_DISTILLED_TEST_RESULTS.md) |

原始 pure 日志曾被审计误导入 job-local `queue.py` 后的失败重试覆盖；原权重和
result 保持完整，历史 `accepted_with_notes` 的限制仍保留。
PPL 审计复核哈希、清单、计数、有限值和 NLL→PPL 算术，没有独立重放前向/logits。
当前 PPL 来自算法 payload 解码为 dense BF16，不能作为 packed kernel 正确性或速度证据。

| 阶段 | 执行源码 revision |
| --- | --- |
| 原始 8B 全模型量化 | `d8a2eff231dee7f0f4c4822179685672d60e10f9` |
| 修复版全模型量化 | `e5f3861c22cd99bbba5cf4bb7bfdf7df3b183de5` |
| 修复版未蒸馏 PPL | `3a3ef5c2aeee260b3b7522171b37536b7fb4d3f5` |
| 蒸馏 | `8b4aa4ac012d06d900bf34ee00701e1a052b0889` |
| 固定 step400 test | `8847049a1d4a1209719e84dd5dee4f6b98fa3081` |

训练 result SHA256：`e7610e122bef9df35b3a70749fd2e7f388bd752d594c69e60da796c73eb937b5`。
最终 test result SHA256：`1b1b4353b200a0a0daa357c3e12d3a35544e6fc2f9d16c427e958644baf750ef`。
小型原始结果、曲线、日志和验收归档在本地 Git-ignored `server_results/`，不提交 GitHub。

## 保存状态与下一阶段

关机前已逐文件校验 36 层 step400 payload，约 2.54 GiB，保存到网络持久卷：
`/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1/`。
Manifest SHA256：`253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec`。
完整模型恢复还依赖原 pinned HF snapshot；最终 optimizer state 留在训练作业目录。
大权重按用户决定不下载本机。服务器关闭由用户确认，未在关闭后重新核验远端存储。
恢复路径和环境锁见 [服务器保存记录](SERVER_SHUTDOWN_READY.md)。

后续以固定 step400 为输入：版本化布局转换 → Linear correctness/计时 → block →
全模型加速。具体约束见 [加速交接](ACCELERATION_HANDOFF.md)。在格式、形状、索引、
dtype、执行路径和环境不变且无数值依赖分支时，后续只更新 scales 不改变运算和访存规模；
仍为新权重重做正确性检查，并保留每次计时绑定的权重及代码版本。

历史 32B 的 test PPL 为 BF16 7.610839、pure 17.744707、hybrid 10.433873，
仅保留为历史算法证据，详见 [README](README.md#historical-accepted-qwen3-32b-result)。
当前不新增 32B 全模型实验。
