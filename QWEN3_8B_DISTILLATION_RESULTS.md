# Qwen3-8B conditioned hybrid 蒸馏验收

2026-09-13：**accepted_validation_only**。完整训练/导出通过，合成与真实验证均改善。
尚无蒸馏后 WT2 test PPL，不将本报告的 validation 数值与旧 test PPL 14.951611 混比。

## 首轮结果

| 指标 | step 0 | step 400 |
|---|---:|---:|
| 合成训练 CE | 4.844360 | 1.765263 |
| 合成训练 feature MSE | 74.447670 | 14.433087 |
| 合成训练归一化总 loss | 2.000000 | 0.558264 |
| 合成验证 CE | 3.658656 | 2.116016 |
| 合成验证 feature MSE | 64.524043 | 14.421741 |
| 合成验证归一化总 loss | 1.621944 | 0.630516 |
| WT2 validation PPL | 15.272183 | 13.670310 |
| WT2 validation feature MSE | 27.899045 | 20.895968 |

合成验证总 loss 降低 61.13%；真实 validation PPL 降低 **10.49%**，feature MSE
降低 25.10%。BF16 teacher validation PPL 在所有监控点均为 10.240814425。

| step | WT2 validation PPL |
|---|---:|
| 0 | 15.272183 |
| 40 | 14.294069 |
| 80 | 14.124159 |
| 120 | 13.925180 |
| 200 | 13.694294 |
| 300 | 13.664953 |
| 400 | 13.670310 |

step 300 比最终值略低，但没有保存该步权重；按冻结规则保留最终 step 400，
不事后挑选最优 step。仍需后续独立 test PPL 来判断标准测试协议下的质量。

## 训练合同与资源

- 父权重：已验收 conditioned hybrid v1；原始 BF16 teacher 固定。
- 400 条生成候选，按 teacher/student logit MSE 降序筛选 200 条；100 条独立验证；
  序列长度 128，训练与验证不重叠。
- 固定初始分母 CE=4.844359564781189、feature=74.44766971588135。
  总 loss 为 CE/初始CE + feature/初始feature；feature 对 36 个 block 平均。
- 400 步 / 2 epochs / batch 1；Adam 1e-6、cosine 至 0、gradient clip 1。
- 1008 个 scale 张量、219875328 个可训练参数；固定 packed signs、选列索引及其他模型参数。
- 完整运行退出码 0。脚本记录阶段耗时 3971.71 秒（66.20 分钟，起点在模型装载/替换后）；
  峰值 torch allocated 26.19 GiB。耗时不是训练 kernel benchmark。
- 真实验证覆盖 128 个 2048-token 块，每个监控点 262016 个预测位置。
- 训练期 test 评估次数 0，无后端/部署任务。

## 审计证据

审计重新检查 400 步编号、每 epoch 数据排列、cosine 学习率、损失归一化算术、
21 个合成监控点及 7 个真实验证点；所有记录有限。复核 total NLL→mean NLL→PPL，
progress 与最终结果/日志一致。重算保存的 scores 的 top200，验证全部候选与验证无重叠。

逐层重哈希父/子 payload，检查全部 252 Linear 的 BF16 解码为有限值；
1764 个张量清单保持一致，四类 scales 各 252 张量更新，所有固定符号/索引逐项相同。
读取最终 optimizer state（weights_only）并验证 1008 份状态、step 400、有限 moments、
scheduler 和数据顺序 RNG。没有重新生成合成序列、重新计算筛选 logits 或独立重放前向。

- 训练源码：`8b4aa4ac012d06d900bf34ee00701e1a052b0889`。
- result SHA256：`e7610e122bef9df35b3a70749fd2e7f388bd752d594c69e60da796c73eb937b5`。
- 审计脚本 SHA256：`1a4e58b238610b9216de41731385a7d0ace9796b42efa38ec7b2f44e1ff05a10`。
- 作业：`/workspace/jobs/qwen3-8b-distill-train-v1/`。
- 结果/验收：`artifacts/result.json`、`artifacts/acceptance.json`。
- 最终权重：`artifacts/payloads/layer-000.safetensors` 至 `layer-035.safetensors`。
- 最终优化器等状态：`artifacts/final_training_state.pt`；无中间 checkpoint/自动恢复入口。
- 本地小备份：`server_results/runpod_hybrid_pcie_2026-09-13/distillation/distillation-accepted-small.tar.gz`。
  包含曲线、数据、配置/provenance、验收和日志；大权重及 optimizer state 保留服务器。
