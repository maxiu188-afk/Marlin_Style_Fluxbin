# Qwen3-8B matched PPL evaluation

2026-09-13：已提交至 RunPod A100-SXM4-80GB，结果待验收。
执行代码：`e45462aa155aeeedf72f98080a03f74852bc9120`。
本地与服务器各通过 48 项单元测试；本地测试不构成 CUDA 数值验证。

## 评估协议

- 模型：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 三组同次顺序执行：BF16 → pure → hybrid-s8；重新实测本机 8B BF16 基线。
- 复用冻结 WikiText-2 token 文件，不重新采样：299,078 tokens，146 个不重叠
  的 2048-token 完整块，298,862 个 scored transitions；丢弃不完整尾块。
- 各块独立 next-token 计分，不计跨块预测；BF16 模型/logits，FP32 交叉熵，
  每 128 个 token 分块求和；batch=1，SDPA，use_cache=false，禁用 TF32。
- 严格复核原始快照、token、验收记录、两组 payload 与内部张量哈希，分别将
  每组全部 252 个 Linear 覆盖为解码后的 BF16 权重；其余权重保留原始 BF16。
- 复用历史计分器与解码逻辑，但不要求复现旧 32B/GH200 的 BF16 PPL。
- 相对本次 BF16 的 PPL 差距 <=5% 才满足质量门槛；>10% 明确阻止部署。
  执行有效与质量达标分别记录，PPL 完成不自动启动 backend 或 distillation。

Token 文件 SHA-256：`252938697260d7f7241f26a05b9822c1a2fd5e9ae0168a87e0d5345d4a111ee0`。
协议 manifest SHA-256：`8b61cbeaba8809b94bc6b7568cb45ede0d941d046c812f57f3156163e42069ae`。
完整冻结字段位于 `configs/evaluation/qwen3_8b_wikitext2_full_hessian_obq_s8_v1.json`。

## 已提交任务

- tmux：`qwen3-8b-ppl-v1`；初始计算 PID：`6391`。
- Job 目录：`/workspace/jobs/qwen3-8b-ppl-v1/`。
- 日志：Job 目录的 `ppl.log`，每 10 个块输出进度。
- 任务状态：`status.json`；结束后生成 `exit-code`。
- 完整启动命令与 Git revision：`launch.json`。
- 结果：服务器 checkout 下 `results/qwen3-8b-ppl-v1/result.json`。
- 每完成一组写入 `result.progress.json`，含计分指标、来源哈希与量化覆盖记录。
- 模型与 token 已在服务器；环境复用 `/opt/fluxbin-venv`，未新增依赖。

任务由 tmux 持久运行，SSH 断开不影响执行。此处为提交记录，最终 PPL 与质量
结论仍需核验结构化结果、计分数、有限性、来源哈希及量化覆盖。执行期间不要更新
服务器代码。完成本轮启动检查后不持续轮询。

在已连接服务器的终端查看状态：

```bash
cat /workspace/jobs/qwen3-8b-ppl-v1/status.json
```

查看最近日志：

```bash
tail -n 30 /workspace/jobs/qwen3-8b-ppl-v1/ppl.log
```

仅在明确需要取消时，单独执行 `kill 6391`。先核实 PID 仍属于本任务，
避免任务已结束后 PID 被复用；此命令不会与查看命令放在同一代码块。
