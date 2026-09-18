# 项目文档索引

项目级说明集中在此目录。文档中的命令及代码、配置、结果路径默认以**仓库根目录**
为工作目录；Markdown 链接相对于所在文档。RunPod 配置旁的维护说明仍放在
[infra/runpod](../infra/runpod/README.md)。

## 先看这些

- [当前进度与交接](CURRENT_HANDOFF.md)：只保留当前有效状态和下一步边界
- [结果总览](RESULTS_OVERVIEW.md)：近期完整模型对照与质量证据
- [历史实验计划与完成状态](EXPERIMENT_PLAN.md)

## M=1 加速与全模型运行

专题总入口：[性能与部署文档](performance/README.md)。

- [GPTQ W3 inline LUT 第一轮运行手册](performance/W3_LUT_INLINE_RUNBOOK.md)
- [GPTQ W3 BF16 舍入修正与 split-G 诊断手册](performance/W3_LUT_BF16_SPLIT_RUNBOOK.md)
- [GPTQ W3 inline LUT A100 PCIe 结果](performance/W3_LUT_INLINE_RESULTS.md)
- [GPTQ W3 inline LUT 完整模型结果与误差归因](performance/W3_LUT_FULL_MODEL_RESULTS.md)
- [GPTQ W3 corrected A100 完整模型结果与复现手册](performance/W3_CORRECTED_FULL_MODEL_RUNBOOK.md)
- [完整模型 logits 数值门的标定](performance/W3_NUMERICAL_GATE_CALIBRATION.md)
- [非 Linear 路径融合：分析、实现与预期](performance/NONLINEAR_FUSION.md)
- [W3 GPU 验证批次运行手册](performance/W3_GPU_VALIDATION_BATCH_RUNBOOK.md)
- [W3 packed M=1 PPL 运行手册](performance/W3_PACKED_M1_PPL_RUNBOOK.md)
- [候选批次及全模型运行手册](performance/M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)
- [加速实验合同与恢复交接](performance/ACCELERATION_HANDOFF.md)
- [M=1 初始准备](performance/M1_ACCELERATION_PREPARATION.md)
- [v2 本地优化设计](performance/M1_V2_LOCAL_OPTIMIZATION.md)
- [Linear / block / 完整模型加速结果](performance/QWEN3_8B_M1_LINEAR_RESULTS.md)

## 服务器环境与恢复

专题总入口：[服务器与运维文档](operations/README.md)。

- [首次 M=1 环境准备记录](operations/RUNPOD_M1_SETUP_RESULTS.md)
- [历史服务器迭代计划](operations/SERVER_ITERATION_PLAN.md)
- [关机与存储交接](operations/SERVER_SHUTDOWN_READY.md)

## Qwen3-8B 重建、质量与蒸馏

专题总入口：[算法与质量文档](quality/README.md)。

| 阶段 | 运行/诊断说明 | 结果 |
|---|---|---|
| Linear 重建 | [运行指南](quality/QWEN3_8B_LINEAR_GUIDE.md) | [Linear](quality/QWEN3_8B_LINEAR_RESULTS.md)、[全模型重建](quality/QWEN3_8B_FULL_RESULTS.md) |
| 初始 PPL | [运行指南](quality/QWEN3_8B_PPL_GUIDE.md)、[诊断](quality/QWEN3_8B_PPL_DIAGNOSIS.md) | [PPL](quality/QWEN3_8B_PPL_RESULTS.md) |
| Hybrid / conditioned | [Hybrid 探针](quality/QWEN3_8B_HYBRID_PROBE_GUIDE.md)、[全模型指南](quality/QWEN3_8B_CONDITIONED_FULL_GUIDE.md) | [全模型重建](quality/QWEN3_8B_CONDITIONED_FULL_RESULTS.md)、[PPL](quality/QWEN3_8B_CONDITIONED_PPL_RESULTS.md) |
| 蒸馏 | [准备与协议](quality/QWEN3_8B_DISTILLATION_PREPARATION.md) | [训练结果](quality/QWEN3_8B_DISTILLATION_RESULTS.md)、[测试结果](quality/QWEN3_8B_DISTILLED_TEST_RESULTS.md) |
| W3/QBB 同码率质量对照 | [冻结配置与运行手册](quality/QWEN3_8B_W3_RATE_DISTORTION_RUNBOOK.md) | [四臂结果与 Case A 决策](quality/QWEN3_8B_W3_RATE_DISTORTION_STATUS.md) |
| Hierarchical-scale W2 | [实现、端点结果与恢复手册](quality/QWEN3_8B_HIERARCHICAL_W2_RUNBOOK.md) | H2.50/H2.875 平坦，两者均未接近 W3 |
| Hierarchical W2 offline rotation | [分级 probe 与正式运行手册](quality/QWEN3_8B_HIERARCHICAL_W2_ROTATED_RUNBOOK.md) | 待运行；layer-0 → 0/17/35 → 人工决定是否全模型 |

## 历史归档

- [早期 8B/32B 执行交接](archive/HANDOFF_HISTORY_THROUGH_2026-09-15.md)：保留验收与来源；历史状态不作为当前指令。
