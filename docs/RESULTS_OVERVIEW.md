# 当前结果总览

更新：2026-09-16。速度比均为原始 BF16 / packed，大于 1 表示加速；
以下是不同硬件与协议的历史结果，不是可直接归因的单因素消融。

## 近期完整模型结果

| 时间 / GPU | 实现与协议 | prompt 0 / 1 速度比 | 状态 |
|---|---|---:|---|
| 09-14 SXM4 | v2/gps8，动态 KV eager v1 | 0.9332x / 0.9562x | 稳定，未加速 |
| 09-15 PCIe | v3/gps4，动态 KV eager v1 | 0.86887x / 不发布 | prompt 1 辅助基线不稳定 |
| 09-15 SXM4 | v4/gps4，动态 KV eager v1 | 0.83063x / 0.83143x | 稳定，未加速 |
| 09-15 PCIe | v5_p1024/gps1，动态 KV eager v1 | 0.87696x / 0.86674x | 稳定，未加速 |
| 09-15 SXM4 | v5_p1024/gps1，prepared eager v2.1 | 不发布 / 不发布 | 波动 6.51–10.11%，超 5% |
| **09-15 SXM4** | **v5_p1024/gps1，sequence Graph v2.1** | **1.380972x / 1.383276x** | **两组稳定，跨度 <0.12%** |
| **09-16 PCIe** | **GPTQ W3 inline，sequence Graph** | **1.488022x / 1.489158x** | **性能稳定；decoded-W3 数值门失败** |

QBB v2.1 Graph 的 32-token 延迟：原始 BF16 467.513 / 467.919 ms，packed
338.539 / 338.269 ms；约 68 → 95 token/s。eager 不稳定导致整体 JSON 为
`completed_unstable`，不能把两组 Graph 的稳定结论写成全协议均通过。
上述数值均在 report-only 策略下测量，不代表 BF16 数值等价。

最新 W3 行使用另一个权重格式、GPU 实例和正式协议，不能与 QBB 行作单因素 kernel
消融。W3 的 32-token packed 延迟为 320.625 / 321.710 ms，原始 BF16 为
477.097 / 479.077 ms；但 packed-vs-decoded W3 logits NRMSE 为 0.01683 / 0.01457，
超过 0.005，因此不是 correctness accepted 结果。

## GPTQ W3 完整模型误差归因

W3 运行覆盖 36 层、252 个 packed Linear；同一 StaticCache 下 prepared wrapper、
Graph、路由、prefill 和 prefix KV 检查通过。prompt 0 的第一个 decode step 在第 0 层
已有 0.001506 NRMSE，第 8 层首次超过 0.005。把每个 packed Linear 的真实输入送入
对应 dense W3 Linear 后，单算子本地 NRMSE 为约 0.00059--0.00264，没有单独异常
shape。

四种真实 shape 的 kernel 输出与 structural FP32 reference 转 BF16 均逐位一致；
FP32 归约顺序差异仅 1e-7 量级。主要差异来自 decoded 路径先物化
`BF16(scale * code)` 权重，而 kernel 先做 FP32 integer dot、再乘 BF16 scale，
最终 BF16 舍入后产生约 0.00065--0.00218 的单层差异，并经完整模型传播放大。
所以本次失败不是 `g_idx/desc_act`、packing、Graph 或 fallback 错误，而是 oracle
与 kernel 的有限精度语义不一致。完整数据、哈希和后续边界见
[W3 完整模型结果](W3_LUT_FULL_MODEL_RESULTS.md)。

## Linear、block 与协议边界

最新孤立 Linear 批次来自上一轮 PCIe：`v5_p1024/gps1` 七个 Graph Linear
合计 124.598 µs，相对同轮 decoded BF16 2.1285x；孤立 eager 为 2.0219x。
这些是 layer-0 七个独立算子的时间和，**不是 block 延迟，也不能外推全部 252 个 Linear**。
同轮 block 数值/路由通过，但 dense 计时不稳定，不发布 block 加速比。
用户明确要求无论 block 速度如何都继续完整模型实验。

最新 v2.1 固定 Qwen3-8B step400、batch1、两个真实 prompt、32 个相同输入 token，
主基线原始 BF16，辅助基线 decoded step400。静态 KV 预填充后恢复，逐步增长位置；
Graph 包含 embedding、全部 36 层、LM head、argmax、LUT 构建与归约。
预填充、reset、capture 和审计在 decode 计时外；三模型同时驻留，轮换 arm/mode，
8 次预热、10 次测量。不是自由生成、TTFT、A8 或 vLLM 结果。

113/113 GPU 测试通过，修正 runner 后专项 7/7 通过；6/6 arm/prompt 的同静态
KV checked/prepared wrapper、Graph/eager 输出精确一致，120 次正式测量的重复检查通过。
packed 两组审计/capture 均覆盖 252 个 Linear，每个执行 32 次、decode dense fallback=0。

跨语义差异独立报告：最新 packed/decoded NRMSE 0.0125357 / 0.0129711，
动态/静态注意力 NRMSE 0.01167–0.01359，均未通过旧阈值。第一次 v2 在 decoded
动态/静态检查处失败、未计时；v2.1 改用相同静态注意力下的 checked wrapper 作
严格参照，保留前次失败记录，不把缓存路径差异解释成包装或 Graph 错误。

完整版本表、每轮原始状态、运行提交与 SHA256 见
[加速详细结果](QWEN3_8B_M1_LINEAR_RESULTS.md)。最新运行代码 `3c996ab`；
本地私有备份 `server_results/runpod_prepared_sxm4_2026-09-15/` 已完成哈希、
源码、120 次样本统计与路由核验，不提交原始大文件到 GitHub。

## Uniform W3 与当前 QBB 的质量/码率决策

2026-09-16 在 RTX PRO 4500 Blackwell 上以 `same-device-quality` 重跑四臂。
所有 arm 使用相同 146 x 2048 WikiText-2 blocks、298,862 positions、dense BF16
评测路径，并覆盖相同 36 layers / 252 Linears / 6,945,767,424 weights。

| Arm | Effective bits/weight | Serialized weight bytes | PPL | Delta vs current QBB |
|---|---:|---:|---:|---:|
| BF16 | 16.000000 | 13,891,534,848 | 9.726488173 | -3.441421766 |
| current QBB | 3.138184 | 2,724,636,672 | 13.167909939 | 0 |
| GPTQ W3A16 g128 sym | 3.154552 | 2,738,847,744 | **11.266114820** | **-1.901795120** |
| QBB FP16 scales | 2.631687 | 2,284,886,016 | 13.168951166 | +0.001041227 |

GPTQ 仅多 0.5216% 存储，PPL 比当前 QBB 低 14.4426%，符合预先定义的
**Case A**。当前约 3.14-bit QBB point 基本被 uniform W3 支配，后续优先
uniform 3-bit backend / solver，不再深挖当前 QBB format 的 kernel。FP16 scales
证明 QBB 可压到 2.6317 bit/weight 而几乎不损失质量，但它是较低码率、较差质量的
另一个 trade-off 点，不改变主决策。详见[专门状态页](QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)。

该证据不等于 A100 `1e-6` 复现，也不是 latency 结果。旧 A100 BF16/QBB anchor
只作 report-only；其跨设备偏移约 0.0015--0.0019 PPL，远小于 1.9018 的同卡
effect size。结果源码 `f979f6b`，summary JSON SHA256 为
`de5e12fcd7c1db5da1952717dd15a01c9c230b34a3d2b254e64d0fc05e88e8cd`。

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

上述加速链条已执行；后续以固定 step400 和 prepared v2.1 为基线评估新候选，
单 block 速度不作为完整模型启动条件。具体约束见 [加速交接](ACCELERATION_HANDOFF.md)。在格式、形状、索引、
dtype、执行路径和环境不变且无数值依赖分支时，后续只更新 scales 不改变运算和访存规模；
仍为新权重重做正确性检查，并保留每次计时绑定的权重及代码版本。

历史 32B 的 test PPL 为 BF16 7.610839、pure 17.744707、hybrid 10.433873，
仅保留为历史算法证据，详见 [README](../README.md#historical-accepted-qwen3-32b-result)。
当前不新增 32B 全模型实验。
