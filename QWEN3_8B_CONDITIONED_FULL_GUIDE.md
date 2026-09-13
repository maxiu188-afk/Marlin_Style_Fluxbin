# 8B hybrid 补偿修复：全模型阶段

本阶段承接已验收的三个 Linear 小对照，扩展到 36 层 / 252 个 Linear。
只运行修复版 hybrid，沿用 C4 256×2048、seed 20260902、g128/s8、50 次 ALS、
1% damping 和逐层校准顺序；蒸馏尚未开始。

## 受控改动

每个 Linear 先对当前目标 W 和 H 执行旧 hybrid，取得列索引，再把相同索引
交给 `quantize_hybrid_conditioned_v1`。两次拟合共享 W/H 和求解器参数。
旧拟合只用于确定列索引，最终保存并传播修复版权重。

前层量化结果变化会自然改变后层输入，因此后层索引不保证等于历史全模型索引；
其选择流程保持旧算法。本阶段不改变层内 Hessian 捕获顺序或迭代预算。
记录旧、新索引哈希并检查一致，验证 packed BF16 回读和断点恢复路径。

冻结模板：`configs/experiments/qwen3_8b_full_hybrid_conditioned_v1.json`。
模板绑定已验收 probe 的结果文件哈希，拒绝 pure、未通过的 probe 或配置漂移。
执行前将旧 full.config 的已验证输入哈希绑定至新模板，生成独立 resolved config。
新产物：`artifacts/qwen3-8b-full-hybrid-conditioned-v1/hybrid_s8/`。
旧模型、旧权重、失败 probe 和已验收 probe 均保留。

## 验收顺序

1. 输入/运行时预检通过后量化，每层原子保存，可从完整层边界恢复。
2. 完成后检查 36 层、252 个 Linear、6945767424 个参数、全部 payload 哈希、
   索引一致和 BF16 重构。全模型旧式 FP32 Hessian 指标仅作近似诊断，
   不用作小 probe 的 FP64 一致性门槛。
3. 权重验收后再绑定新权重执行原 WT2 PPL 协议，与 BF16 9.724945 和旧 hybrid
   16.142104 比较。全模型输出误差或小 Linear 降幅不能代替 PPL。
4. PPL/后端/蒸馏均不由量化脚本自动触发。

本地和服务器均通过 59 项测试，包括新 full 路由与 probe 的数值一致性、固定列与 packed
回读，以及 pure/缺失 probe 拒绝检查。服务器 GPU/依赖检查通过。


## 本次提交

- 源码：`e5f3861c22cd99bbba5cf4bb7bfdf7df3b183de5`。
- 服务器：`root@213.173.105.9:41889`，A100 80GB PCIe。
- tmux：`qwen3-8b-hybrid-conditioned-v1`，初始计算 PID 1698。
- 作业目录：`/workspace/jobs/qwen3-8b-hybrid-conditioned-v1/`。
- 日志：作业目录 `hybrid.log`；PID：`hybrid.pid`；退出后生成 `exit-code`。
- resolved config 和最终结果：checkout 的
  `results/qwen3-8b-full-hybrid-conditioned-v1/{full.config.json,hybrid_s8.json}`。
- 启动记录：`/workspace/jobs/qwen3-8b-hybrid-conditioned-v1.launch.json`。

任务已完成并通过完整性/重构验收：36/36 层，252 个 Linear，退出码 0。
详见 [验收结果](QWEN3_8B_CONDITIONED_FULL_RESULTS.md)。PPL 尚未运行。
执行期间保持远端代码版本不变。每层完成会输出 `FLUXBIN_FULL_LAYER_COMPLETE`。

以下命令均在服务器执行。只读查看日志：

```bash
tail -n 30 /workspace/jobs/qwen3-8b-hybrid-conditioned-v1/hybrid.log
```

以下 PID 仅为历史记录；任务已结束，不再用它控制进程。历史查看命令：

```bash
ps -p 1698 -o pid,etime,stat,args
```

任务已结束，无需执行取消命令。


## 修复版 PPL 提交

全模型权重验收通过后，用户授权继续 PPL。冻结配置为
`configs/evaluation/qwen3_8b_wikitext2_conditioned_hybrid_v1.json`，独立入口
`scripts/run_qwen3_8b_conditioned_hybrid_ppl.py` 复用原 scorer/decoder。
本地和服务器均通过 61 项测试。

- 源码：`3a3ef5c2aeee260b3b7522171b37536b7fb4d3f5`。
- tmux：`qwen3-8b-conditioned-ppl-v1`；初始 PID 2243。
- 作业：`/workspace/jobs/qwen3-8b-conditioned-ppl-v1/`，含 `status.json`、
  `ppl.log`、`ppl.pid`、`exit-code`、`launch.json`。
- 输出：checkout 的 `results/qwen3-8b-conditioned-ppl-v1/result.json`。
- 状态：独立会话已启动，输入/权重预检正在执行，尚无 PPL 结果。

本轮按相同顺序先 BF16 后修复版 hybrid，使用原 WT2 固定 token、146 块、
298862 次预测、FP32 CE、SDPA、禁用 cache/TF32 和原质量门槛。
原 hybrid 16.142104 是历史对照，本轮不重跑 pure 或旧 hybrid。
结果需区分执行有效性与质量通过；不会触发后端或蒸馏。

服务器只读查看：

```bash
tail -n 30 /workspace/jobs/qwen3-8b-conditioned-ppl-v1/ppl.log
```

只读状态：

```bash
cat /workspace/jobs/qwen3-8b-conditioned-ppl-v1/status.json
```

如必须停止，先确认 `ps -p 2243 -o pid,args` 仍对应本次任务，再单独执行
`kill -TERM 2243`。不要把取消命令和查看命令一起粘贴。
