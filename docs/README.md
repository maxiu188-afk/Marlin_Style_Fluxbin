# 项目文档索引

项目级说明集中在此目录。文档中的命令及代码、配置、结果路径默认以**仓库根目录**
为工作目录；Markdown 链接相对于所在文档。RunPod 配置旁的维护说明仍放在
[infra/runpod](../infra/runpod/README.md)。

## 先看这些

- [当前进度与交接](CURRENT_HANDOFF.md)
- [结果总览](RESULTS_OVERVIEW.md)
- [实验计划](EXPERIMENT_PLAN.md)

## M=1 加速与全模型运行

- [候选批次及全模型运行手册](M1_CANDIDATES_FULL_MODEL_RUNBOOK.md)
- [加速工作交接](ACCELERATION_HANDOFF.md)
- [M=1 初始准备](M1_ACCELERATION_PREPARATION.md)
- [v2 本地优化设计](M1_V2_LOCAL_OPTIMIZATION.md)
- [M=1 Linear 已测结果](QWEN3_8B_M1_LINEAR_RESULTS.md)

## 服务器环境与恢复

- [首次 M=1 环境准备记录](RUNPOD_M1_SETUP_RESULTS.md)
- [后续服务器迭代计划](SERVER_ITERATION_PLAN.md)
- [历史关机与存储交接](SERVER_SHUTDOWN_READY.md)

## Qwen3-8B 重建、质量与蒸馏

| 阶段 | 运行/诊断说明 | 结果 |
|---|---|---|
| Linear 重建 | [运行指南](QWEN3_8B_LINEAR_GUIDE.md) | [Linear](QWEN3_8B_LINEAR_RESULTS.md)、[全模型重建](QWEN3_8B_FULL_RESULTS.md) |
| 初始 PPL | [运行指南](QWEN3_8B_PPL_GUIDE.md)、[诊断](QWEN3_8B_PPL_DIAGNOSIS.md) | [PPL](QWEN3_8B_PPL_RESULTS.md) |
| Hybrid / conditioned | [Hybrid 探针](QWEN3_8B_HYBRID_PROBE_GUIDE.md)、[全模型指南](QWEN3_8B_CONDITIONED_FULL_GUIDE.md) | [全模型重建](QWEN3_8B_CONDITIONED_FULL_RESULTS.md)、[PPL](QWEN3_8B_CONDITIONED_PPL_RESULTS.md) |
| 蒸馏 | [准备与协议](QWEN3_8B_DISTILLATION_PREPARATION.md) | [训练结果](QWEN3_8B_DISTILLATION_RESULTS.md)、[测试结果](QWEN3_8B_DISTILLED_TEST_RESULTS.md) |
