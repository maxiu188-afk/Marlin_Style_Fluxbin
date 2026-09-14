# M=1 候选批次与全模型运行准备

2026-09-14，本地准备；未连接服务器，未编译或测试新 CUDA 候选。保留 v1
负面结果。vLLM 接口保留，未接入。当前目标是一次上机完成一批有界实验，
安装/测试本身耗时不长，暂不投入复杂镜像自动化。

## 固定候选

配置：`configs/acceleration/m1_candidates_v1.json`。6 组包含 1 个基线、5 个候选。

| ID | kernel | 每 warp 输出行 | groups_per_split | 要检验的问题 |
|---|---|---:|---:|---|
| baseline | v1 | 1 | 8 | 保留同轮基线 |
| swizzle | v2_r1 | 1 | 8 | swizzle 与 shared int32 lookup |
| rows2 | v2_r2 | 2 | 8 | 两行复用 |
| rows4 | v2 | 4 | 8 | 四行复用 |
| rows4_split4 | v2 | 4 | 4 | 增加 K 分片 |
| rows4_split16 | v2 | 4 | 16 | 减少 K 分片 |

全部为 layer 0 的七个真实 Linear、BF16，分别 eager/Graph，共 12 个 trial。
每 trial 最长 300 秒，超时或正确性失败即停止并保存日志；不稳定 timing 保留，
不提供可接受的速度结论。每项固定 warmup=20、repeats=100、rounds=7。
非 v1 候选同时对照同 split 的 v1 和 dense oracle；split 改变可改变跨组求和次序，
只要求同 split 的候选/v1 完全一致，所有项仍必须满足 dense 数值门槛。
ptxas register/spill 信息进入构建日志。v1 CUDA 源码保持不变。

## 上机前与恢复

先确认 Git 推送成功。服务器通过 Git fetch / clean fast-forward 同步；保留网络卷
及其模型、step400 payload、结果、兼容编译缓存。按
`RUNPOD_M1_SETUP_RESULTS.md` 和 `ACCELERATION_HANDOFF.md` 恢复环境；源码已变化，
必须另存新环境记录，不复用旧 source hashes。基础镜像 digest 仍未知。

以下命令均在服务器仓库根目录、已激活项目环境执行。预先设置并导出：
`FLUXBIN_ARTIFACT_ROOT`（含 manifest.json 的已接受 step400 目录）、
`FLUXBIN_SNAPSHOT_ROOT`（固定 revision 的完整模型目录）、
`FLUXBIN_RUN_ROOT`（本次新的持久化结果目录）。不在这里硬编码机器路径。

```bash
python scripts/record_acceleration_environment.py \
  --output-dir "$FLUXBIN_RUN_ROOT/environment" --phase recreated \
  --image-reference 'runpod-default-unresolved' --require-cuda --build-smoke
export FLUXBIN_NEW_ENVIRONMENT_JSON="$FLUXBIN_RUN_ROOT/environment/environment.json"
```

`--build-smoke` 会编译全部四个 kernel variant，测试 FP16/BF16、行/split 尾部、
workspace 覆写、非默认 stream、Graph 及同 split 的 v1 精确一致性。没有 CUDA
不能标记 ready；新 smoke 失败不得继续。基础镜像可识别时填真实 tag/digest。

## 分阶段明确启动

长任务在 tmux 会话内执行，完成启动后可 detach，避免依赖 SSH 前台连接。
各阶段单独启动、保存完整日志和退出码，不自动晋级。

```bash
tmux new-session -s fluxbin-m1-batch
# 在该 tmux 会话内执行；Ctrl-b d 可 detach。
timeout 3700s python scripts/run_m1_candidate_batch.py \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output-dir "$FLUXBIN_RUN_ROOT/candidates" \
  > "$FLUXBIN_RUN_ROOT/candidate-batch.log" 2>&1
printf '%s\n' "$?" > "$FLUXBIN_RUN_ROOT/candidate-batch.exit"
```

检查：`tmux attach -t fluxbin-m1-batch`，或读取 batch.json / 对应日志 / exit。
取消：在会话内 Ctrl-C；外层 timeout 限制整批上限。不要仅凭 exit=0 接受性能。
每个 trial 的 result/log hash 保存在 batch.json；batch 完成不代表每项 timing 稳定。

查看七格数值、稳定性和同轮速度比后，明确选择一份合格 Linear JSON。
将其路径设置为 `FLUXBIN_SELECTED_LINEAR_JSON`，然后在持久会话中运行：

```bash
timeout 600s python scripts/run_m1_block_probe.py \
  --snapshot-root "$FLUXBIN_SNAPSHOT_ROOT" --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --linear-result "$FLUXBIN_SELECTED_LINEAR_JSON" \
  --output "$FLUXBIN_RUN_ROOT/block.json" > "$FLUXBIN_RUN_ROOT/block.log" 2>&1
```

block 自动继承所选 kernel/split；检查七层替换、三次数值/repeat gate 和七轮稳定
计时。它仍只是 layer 0、synthetic hidden、空 KV cache 的完整 block 探针。
通过并审阅后，才能明确启动全模型：

```bash
timeout 1800s python scripts/run_qwen3_8b_full_m1_trial.py \
  --snapshot-root "$FLUXBIN_SNAPSHOT_ROOT" --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --block-result "$FLUXBIN_RUN_ROOT/block.json" \
  --output "$FLUXBIN_RUN_ROOT/full-model.json" > "$FLUXBIN_RUN_ROOT/full-model.log" 2>&1
```

失败不自动重跑或扩展；保留已有 JSON/log，检查退出码。外层硬超时可能使 JSON
仍为 running，应结合退出码判定，不当作完成。实验结束后备份小型结果与 hash。

## 全模型测量合同

固定配置 `configs/acceleration/qwen3_8b_full_m1_v1.json`：Qwen3-8B，36 个 block、
252 个目标 Linear，batch=1，两个固定 prompt，每个 32 次 cached decode，
每 arm/prompt 1 次 warmup、3 次测量，seed 固定。保留额外的 prefill prediction，
因此记录 33 个 next-token prediction。固定长度运行不因 EOS 提前结束。

三条路径顺序加载、释放，避免同时保留三个模型：step400 decoded BF16、原始 BF16、
step400 packed。所有路径使用 step400 decoded 首次运行产生的同一 continuation，
保证 KV 内容/上下文可比。packed 对照 decoded 检查 logits 数值门槛、greedy token
完全一致、最大 logprob 差 <=0.05；失败保存诊断，不能声称正确或加速。
原始 BF16 是独立性能参照，权重不同，不要求与 step400 输出一致。

- 先校验固定 snapshot 的文件 hash、全部 36 层 payload 和匹配源码/环境/block gate。
- 每个 packed Linear 恰好 prefill fallback 1 次、M=1 kernel 32 次；decode 禁止 fallback。
- KV cache 必须增长至 prompt length+32。attention、norm、residual、embedding/lm_head
  仍由 HF 执行，接口不改，未接入 vLLM、TP 或连续批处理。
- 单列模型载入/转换耗时、峰值与常驻 allocated memory、prefill wall/device、decode
  wall/device、每 token device 时间、吞吐。每条路径 decode wall/device 相对极差
  <=10% 才能报告稳定速度比。该短协议是研究性 smoke/性能试验，不代表长上下文
  或 serving 吞吐，也不代替 PPL/质量验证。
- packed prefill 按需重建 dense 权重，开销计入 prefill；载入方式为先加载 dense BF16
  再替换，并非直接低内存 packed loader。所有路径的 logits 保留和 CUDA event
  采样有共同的测量开销，CPU 拷贝在计时结束后执行。

本地验证：81 项 unittest，79 通过、2 CUDA skip；包含真实 tiny Qwen3 的合成权重
缓存/全前缀一致性测试。完整 8B、CUDA 编译、数值稳定性和所有速度均待服务器验证。
