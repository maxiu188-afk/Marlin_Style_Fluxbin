# 修复版 hybrid 蒸馏准备

状态：核心组件与候选配置完成，本地测试通过；未生成数据、未训练、未运行 GPU smoke。
学生父版本为验收后的 conditioned hybrid（WT2 test PPL 14.951611），教师为同一
Qwen3-8B BF16 snapshot b968826d9c46dd6066d109eabc6255188de91218。

## QBB-New 对应关系与 loss

已阅读 QBB-New 的 `src/qbb_rank1/distillation.py`、
`scripts/run_qwen3_rank_one_scale_distillation_sample200.py` 和 joint shared-column
sample200 配置；参考文件 SHA256 固定记录在本项目候选配置中。
QBB-New 训练用 hard next-token CE，teacher/student logit MSE 用于筛样，
不是训练 logit loss；本准备沿用其训练定义。

```text
L_token = CE(student_logits[:, :-1, :], input_ids[:, 1:])
L_feature = mean_over_36_blocks(mean((student_block_output - teacher_block_output)^2))
L_total = L_token / initial_train_CE + L_feature / initial_train_feature_MSE
```

CE 在 token/vocab 上按 FP32 计算；feature MSE 在 FP32 计算，取每个完整
Transformer block 的输出（含残差路径、最终模型 norm 之前），不含 embedding，
逐层平均。教师 eval、冻结、no_grad；两项权重均为 1。
初始分母在开始训练前对固定的 200 条训练序列求平均，只计算一次，持久化保存。
不得每步重算分母，也不用 validation/test 拟合分母。等长无 padding 序列；
若未来加入 padding，必须补充显式 mask，不能直接复用当前 reduction。

## 学生参数与实现

`src/fluxbin_style/distillation.py` 提供：

- `HybridScaleLinear`：固定 global/refinement packed signs 和选列索引，训练四类 FP32
  scales；保留现有 decoder 的算术顺序，支持导出同格式 payload。
- `distillation_loss`：下一 token CE、36 层 feature MSE、冻结初始归一化。
- `forward_losses`：teacher/student post-block hooks、冻结教师检查、异常时移除 hooks。

252 个 Linear 共 1008 个可训练 scale 张量、219875328 个 FP32 参数。
embedding、norm、lm_head、bias 全部应冻结。学生替换器需先冻结父模型，再安装这些
可训练模块，并检查实际 trainable inventory；不能直接优化一个未冻结的 HF 模型。

使用非 reentrant checkpoint 在 backward 重构临时 dense 权重，固定符号以 packed
格式常驻。CPU 小模型已验证 forward/grad 与普通 differentiable decoder 一致、
仅 scales 更新、导出/重新加载 BF16 输出一致、teacher 无梯度及 hooks 清理。
这不构成 A100 显存可行性或训练吞吐证据。

## 候选首轮预算

配置：`configs/experiments/qwen3_8b_conditioned_distillation_sample200_v1.json`。
沿用参考预算：teacher 生成 400 条长度 128 的候选，按 teacher/student logit MSE
筛选 200 条；另外独立生成 100 条验证序列，检查 train/validation 不相交。
2 epochs / 400 steps / batch 1；Adam lr 1e-6、cosine→0、clip norm 1。
监控训练/合成验证 loss 与固定 WT2 validation；保留最后一步，不按 test 选模型。
训练期间不访问 WT2 test，不自动触发测试集评估。
新 seed 20260913，合成验证 seed 20260914；它们是本实验候选配置，不声称已运行。

## 开训前剩余集成与验收

1. 集成父结果/验收/source/全部 payload 哈希预检，以及 252 Linear 替换和参数冻结。
2. 集成参考 synthetic generation/filter、去重与固定 validation 协议、初始分母计算。
3. 完成训练循环、持久化 optimizer/scheduler/RNG/分母/数据顺序和最终 scales 导出；
   使用新的输出目录，旧父权重不可覆盖。参考 sample200 不保存中间 scale checkpoint；
   若增加恢复 checkpoint，需作为显式配置扩展，不能混用旧规则。
4. 在 A100 80GB PCIe 上先做单 batch 前向/反向与一步更新 smoke；核对 step0
   BF16 权重回读、teacher 无梯度、1008 张量可训练、固定符号/索引哈希和峰值显存。
5. smoke 通过后才提交首轮训练。质量未知，不保证蒸馏收益。

本地共 64 项测试通过；当前只完成准备组件，尚无可直接提交的完整训练入口。
