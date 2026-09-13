# 修复版 hybrid 蒸馏准备

状态：完整 smoke/train 入口已接通，本地和服务器各 65 项测试通过；GPU smoke 已通过，正式首轮任务已提交。
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

## 完整入口与执行门槛

入口 `scripts/run_qwen3_8b_hybrid_distillation.py` 支持 `--mode smoke` 和 `--mode train`。
预检绑定父验收、父结果、source manifest、全部层 payload，以及原始 snapshot 文件哈希；
不读取 WT2 test token。学生替换后检查 252 Linear 和 1008 个可训练 scale 张量。

`distillation_reference.py` 包含从已记录 SHA256 的 QBB-New 实现提取的生成、筛选和
validation 函数。候选按 logit MSE 从大到小取 200 条；重复或训练/验证交叉直接报错。
WT2 validation 来自配置固定的 Salesforce/wikitext revision，不使用本轮 test 结果选模型。

初始分母、训练顺序/每步 loss、合成与真实验证曲线、数据 hash 和最后 36 层 packed
payload 独立保存。无中间 checkpoint；最终保存 optimizer/scheduler/RNG/分母状态，
作为明确的配置扩展。当前没有自动断点恢复入口，失败不会覆盖已有输出。
最终导出逐层回读检查；训练前后固定符号/索引哈希必须一致。

## GPU smoke 与训练提交

执行源码：`8b4aa4ac012d06d900bf34ee00701e1a052b0889`。
smoke 作业：`/workspace/jobs/qwen3-8b-distill-smoke-v1/`，初始 PID 2745。
日志 `run.log`，状态 `status.json`，结果 `artifacts/result.json`。
smoke 使用随机输入，不生成或筛选正式训练数据，不覆盖父权重。
覆盖 step0 252 Linear BF16 解码一致、一轮反向/Adam 更新、冻结教师、符号和索引不变、
四类 scales 更新，以及 batch4 / batch32 / 2048-token 的监控/生成/验证形状。

训练必须提供同一源码对应的 passed smoke result；任何代码/配置改动会要求重新 smoke。
首轮训练输出独立目录，原始和修复版父权重继续保留。
任务有独立 tmux 包装及退出码；正式训练提交状态将在此更新。

单元测试包括梯度对照、父模型替换/冻结、无输入梯度时 scales 仍接收梯度、固定缓冲区、
packed 导出回读、CE shift、层均值与归一化、教师无梯度及 hooks 清理。


## 已验收 smoke / 已提交训练

smoke 退出码 0，结果 `passed`：252 个 Linear 初始 BF16 解码精确一致，
四类 scales 各 252 个张量发生更新，teacher 无梯度，符号/索引哈希不变。
随机 smoke 输入的 CE 13.183447、feature MSE 128.077835、归一化总 loss 2.0；
该总值由初始归一化定义，不能作为训练改善证据。梯度范数 161.220032（clip 前），
测试峰值 allocated 26.28 GiB。数据生成/真实训练的峰值仍以最终运行记录为准。

本地记录：`server_results/runpod_hybrid_pcie_2026-09-13/distillation/smoke.json`。

正式任务已提交到同一 A100 PCIe：

- tmux：`qwen3-8b-distill-train-v1`；初始 PID **2839**。
- 作业目录：`/workspace/jobs/qwen3-8b-distill-train-v1/`。
- 源码：`8b4aa4ac012d06d900bf34ee00701e1a052b0889`；与通过的 smoke 相同。
- 日志/状态：`run.log`、`status.json`、`pid`、`exit-code`。
- 结果：`artifacts/result.json`；曲线 `artifacts/progress.json`；
  数据 `artifacts/synthetic.safetensors`；最终权重 `artifacts/payloads/`。
- WT2 validation 从固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`
  下载至持久缓存，训练记录其原始文件、文本、token 和 block 哈希。
- 最后观察：进程已独立启动，正在预检；尚未报告生成或训练完成。

服务器只读查看：

```bash
tail -n 30 /workspace/jobs/qwen3-8b-distill-train-v1/run.log
```

服务器只读状态：

```bash
cat /workspace/jobs/qwen3-8b-distill-train-v1/status.json
```

如需停止，先确认 `ps -p 2839 -o pid,args` 仍指向本次训练，再单独执行
`kill -TERM 2839`。取消命令不要与查看命令一起粘贴。当前无自动恢复功能；
中断前请注意未完成步骤不会生成中间 scales checkpoint。独立 tmux 可在断开 SSH 后运行。
