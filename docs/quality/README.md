# 算法与质量文档

本目录只收录量化表示、重建、PPL 和蒸馏证据。性能、kernel 和
完整模型 decode 请转到 [performance](../performance/README.md)；当前项目结论
仍以 [总览](../RESULTS_OVERVIEW.md) 和 [交接](../CURRENT_HANDOFF.md) 为准。

## 当前决策入口

- [Hierarchical-scale W2 结果与运行手册](QWEN3_8B_HIERARCHICAL_W2_RUNBOOK.md)：H2.50/H2.875 端点平坦，负面证据。
- [Offline rotation 分级 probe 与正式运行手册](QWEN3_8B_HIERARCHICAL_W2_ROTATED_RUNBOOK.md)：先过 layer-0 和 0/17/35 配对门，正式全模型不自动启动。
- [W3/QBB 四臂结果](QWEN3_8B_W3_RATE_DISTORTION_STATUS.md)：Case A，当前 QBB point 被 uniform W3 支配。
- [W3/QBB 冻结运行手册](QWEN3_8B_W3_RATE_DISTORTION_RUNBOOK.md)。

## Qwen3-8B 质量链条

- Linear 与全模型重建：[Linear 指南](QWEN3_8B_LINEAR_GUIDE.md)、[Linear 结果](QWEN3_8B_LINEAR_RESULTS.md)、[全模型结果](QWEN3_8B_FULL_RESULTS.md)。
- 初始 PPL：[运行指南](QWEN3_8B_PPL_GUIDE.md)、[诊断](QWEN3_8B_PPL_DIAGNOSIS.md)、[结果](QWEN3_8B_PPL_RESULTS.md)。
- Hybrid/conditioned：[探针](QWEN3_8B_HYBRID_PROBE_GUIDE.md)、[全模型指南](QWEN3_8B_CONDITIONED_FULL_GUIDE.md)、[重建结果](QWEN3_8B_CONDITIONED_FULL_RESULTS.md)、[PPL 结果](QWEN3_8B_CONDITIONED_PPL_RESULTS.md)。
- 蒸馏：[准备与协议](QWEN3_8B_DISTILLATION_PREPARATION.md)、[训练结果](QWEN3_8B_DISTILLATION_RESULTS.md)、[最终 test](QWEN3_8B_DISTILLED_TEST_RESULTS.md)。
