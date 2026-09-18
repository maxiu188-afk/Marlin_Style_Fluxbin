# 性能与部署文档

本目录收录 M=1 kernel、Linear/block/全模型 decode、数值门、融合和
packed PPL 路径。质量表示与蒸馏请转到 [quality](../quality/README.md)。

## 当前主入口

- [完整模型结果与误差归因](W3_LUT_FULL_MODEL_RESULTS.md)。
- [GPU 验证批次](W3_GPU_VALIDATION_BATCH_RUNBOOK.md)。
- [非 Linear 融合](NONLINEAR_FUSION.md)。
- [数值门标定](W3_NUMERICAL_GATE_CALIBRATION.md)。
- [Packed M=1 PPL](W3_PACKED_M1_PPL_RUNBOOK.md)。
- [Linear/block/全模型加速总记录](QWEN3_8B_M1_LINEAR_RESULTS.md)。

## 复现与历史路径

- [加速实验合同](ACCELERATION_HANDOFF.md)、[候选批次手册](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)。
- [M=1 初始准备](M1_ACCELERATION_PREPARATION.md)、[v2 本地优化](M1_V2_LOCAL_OPTIMIZATION.md)、[dynamic-warp 分析](M1_DYNAMIC_WARP_INSTRUCTION_ANALYSIS.md)。
- [W3 LUT 早期实现计划](W3_LUT_CODEX_PLAN.md)：历史设计记录，当前结论以本目录的结果文档为准。
- [Inline 运行手册](W3_LUT_INLINE_RUNBOOK.md)、[inline 结果](W3_LUT_INLINE_RESULTS.md)、[BF16/split-G 诊断](W3_LUT_BF16_SPLIT_RUNBOOK.md)、[corrected 路线](W3_CORRECTED_FULL_MODEL_RUNBOOK.md)。
