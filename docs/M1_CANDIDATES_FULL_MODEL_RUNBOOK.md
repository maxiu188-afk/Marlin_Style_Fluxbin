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

`--build-smoke` 会编译全部五个 kernel variant，测试 FP16/BF16、行/split 尾部、
workspace 覆写、非默认 stream、Graph；v2 要求同 split 的 v1 精确一致性，v3
检查 dense 数值容差与自身重复一致性。没有 CUDA
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

## 本地离线汇总与流程检查

服务器结果备份后，在已安装项目环境的本地仓库根目录执行（不需要 GPU）：

```bash
python scripts/summarize_m1_candidates.py \
  --batch-dir "$FLUXBIN_RUN_ROOT/candidates" \
  --output-dir "$FLUXBIN_RUN_ROOT/candidate-summary"
```

生成 summary.json 与 summary.md，固定展示全部 84 格（12 trial × 7 Linear），
保留未运行、失败、hash 不匹配及不稳定项。检查 batch/config/result/log hash、
源码/环境/runner/payload 一致性和输入 hash；从七轮原始样本重算 median、稳定性及
速度比。eager/Graph 分开给出每个 shape 最快的合格实测配置，不自动选定全模型
kernel 或启动下一阶段。缺失/无效格不提供速度比，summary 完成本身不代表实验验收。

新增本地流程检查使用合成 tiny Qwen3 的 36 层，实际调用替换逻辑确认 252 个
Linear 的 kernel/split/fallback 传递；仅 mock 8B 架构尺寸门槛与 payload IO。
另验证最后一层不匹配时不会先替换前面层，以及 CPU prefill 后严格 decode 拒绝
CPU 输入、异常后恢复全部 fallback 策略。这不是 GPU packed forward 验证。
生产替换入口已将全部目标 Linear 与 payload 合同检查前移到首次替换之前。

本轮本地套件 85 项：83 通过、2 CUDA skip；不连接服务器，不接入 vLLM。

## 用户指定的性能优先重跑

首轮因全模型数值差异主动退出后，用户明确允许保留差异并完成性能测量。
使用相同的全模型命令，追加 `--numerical-policy report-only` 并另选新 output 文件。
原始容差、输入、重复次数及 kernel 均不改；误差与 token 差异继续写入 JSON。
coverage、相同输入上下文和有限值检查仍保留。默认不传此参数时仍为 strict。
结果同时记录 `all_numerical_checks_passed` 和 `all_timings_stable`，数值不通过而
稳定测完的状态为 `completed_with_numerical_differences`。主性能基线为 original BF16。

## kernel v3 上机批次

新主循环说明见 [Marlin/MMA 适配](M1_V2_LOCAL_OPTIMIZATION.md)。
本次改动只在本地；原 v1 候选配置和历史结果冻结不变。新配置包含 v1、v2 对照及
v3 的 gps=4/8/16/1024，仍为 6 组、12 个 eager/Graph trial。源码已改变，上机须先生成
新的 source-bound environment（`--build-smoke` 会包含 v3 GPU 检查）。

```bash
python scripts/run_m1_candidate_batch.py \
  --config configs/acceleration/m1_marlin_candidates_v1.json \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output-dir "$FLUXBIN_RUN_ROOT/marlin-candidates"

python scripts/summarize_m1_candidates.py \
  --config configs/acceleration/m1_marlin_candidates_v1.json \
  --batch-dir "$FLUXBIN_RUN_ROOT/marlin-candidates" \
  --output-dir "$FLUXBIN_RUN_ROOT/marlin-summary"
```

仍按本手册的持久会话/超时方式运行。先看 correctness、ptxas register/spill 和
同轮 dense/v1/v3 数据；不把本地测试通过写成 GPU 加速。后续 full-model 的主要速度
基线是原始 BF16，数值差异可按既定 report-only 策略记录。

## v3 瓶颈诊断准备（2026-09-15，尚未执行）

后续用户已取消诊断：本节入口保留但暂停，不继续准备全模型 profiler、不启动采样。
新的候选方向与静态归因边界见 [优化说明](M1_V2_LOCAL_OPTIMIZATION.md)。

本次先诊断未改动的 v3/gps4，不同时改 kernel、算法舍入或量化格式。
入口 `scripts/profile_m1_bottleneck.py` 使用 layer-0 已接受 payload，固定 BF16、
M=1、seed=20260914、gps4，选择 k_proj、q_proj、gate_proj、down_proj 四种形状。
不需要完整模型 snapshot，也不重复全模型 shard 校验。环境仍必须 source-matched。

1. 普通进程执行 timing 模式：dense/v1/v2/v3 同轮 Graph，warmup20、100 repeats、
   7 rounds、10% 稳定性门槛；所有对照用同一 gps4，不冒充上轮 gps8 基线。
2. 独立进程执行 profile 模式：先完成 JIT、dense 重建、数值与重复检查、20 次
   warmup，再用 CUDA profiler start/stop 包住三次指定 arm 调用。
   先采四个形状的 v3 与 dense，共八份报告；每进程最多采六次 kernel launch。
3. 分别看 v3 主 kernel 和 finish reduction：DRAM/L2 实际流量及命中、shared
   bank conflicts、SM/Tensor pipe 利用率、eligible warps、issue rate、stall、
   指令类型和 source/SASS。区分 FP32 重建/转换、地址与位操作、barrier 等候。
4. profiler 内的时间不用于速度比，不能把分离微基准耗时直接相加或相减当作
   融合 kernel 的组成成本。只有 counters/stall/指令共同支持才提出归因。

先检查 `ncu --version`、`ncu --list-sets`，确认 detailed set 存在，并检查 GPU
counter 权限；若权限拒绝，记录原始错误，不自行更改主机驱动安全设置。

```bash
python scripts/profile_m1_bottleneck.py --mode timing --module mlp.gate_proj \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output "$FLUXBIN_RUN_ROOT/gate-timing.json"

ncu --profile-from-start off --set detailed --launch-count 6 \
  --cache-control none --clock-control none \
  -o "$FLUXBIN_RUN_ROOT/gate-v3" \
  python scripts/profile_m1_bottleneck.py --mode profile --arm v3 --module mlp.gate_proj \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output "$FLUXBIN_RUN_ROOT/gate-v3-capture.json"
```

其余 module 分别替换为 self_attn.k_proj、self_attn.q_proj、mlp.down_proj，
dense capture 使用 `--arm dense`，每次使用新 output 名。持久 tmux 中单任务串行，
每进程 timeout 300s，失败保存报告后检查，不启动无界 profile。
默认不主动冲刷缓存，报告必须注明 warmup/replay 条件：重复单 Linear 的 packed
权重可能命中 L2，不能把它等同全模型 streaming。若缓存状态妨碍解释，再单独
补充明确标记的 cold-cache profile（非正式速度），不混入本次固定 timing。

判读顺序：若归并占比高，优化 split/归并；若主 kernel 的 memory pipe/DRAM
达到高利用且 load stall 主导，处理流量/布局；若 Tensor pipe 低、FP32/整数指令
与依赖 stall 主导，优化重建/fragment 生成；若 barrier/shared 冲突突出，改流水。
这些是待检验假设，不是已经测出的瓶颈。暂不加入会被编译器删除计算或显著改变
寄存器压力的“空解码/空 MMA”变体来声称精确归因。

Nsight replay、cache control 与 profiling overhead 的边界见
[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

## 持久环境复用（本地入口已准备，服务器待验证）

`scripts/prepare_persistent_runtime.py` 在持久卷创建新的 Linux venv，而非搬迁
现有 venv。必须使用模板基础 Python，先确认网络卷真实挂载并恢复 nvcc/c++/ninja
的 PATH。模板 PyTorch 仍经 system-site-packages 复用，Python/CUDA/系统工具仍在
容器镜像；持久盘并不能代替基础镜像。

```bash
python scripts/prepare_persistent_runtime.py --persist-root "$PERSIST_ROOT" \
  --output-dir "$FLUXBIN_RUN_ROOT/runtime-restore" \
  --image-reference runpod-default-unresolved
source "$FLUXBIN_RUN_ROOT/runtime-restore/runtime.sh"
```

首次在 environments/<fingerprint>/venv 安装锁定依赖和 editable 项目；随后兼容
启动跳过安装，只核对 pip check、锁定版本和导入。指纹包括 Python、架构、libc、
基础 package 版本、torch/CUDA、编译器、GPU capability、lock、项目依赖元数据及
固定解释器/仓库路径。变化就建新环境，不修改旧环境；未完成安装的目录不会被
当作成功缓存。不同进程不得同时修改同一已准备 venv。

生成 runtime.json 记录总恢复和导入时间，runtime.sh 设置 venv、nvcc 和兼容
extension cache。首次新命名空间仍会编译，不承诺立即复用旧 namespace 的二进制。
源码修改触发必要 JIT；完整 GPU 测试只在首次/相关改动后运行，重复开机仍需
新的环境记录和小型真实 CUDA 调用确认，不能仅凭 venv 存在宣称 ready。

网络盘大量小文件访问可能拖慢安装/导入/JIT，且运行期间仍需该挂载可访问。
完成 warmup 后纯 GPU kernel 并不读取 venv 文件；但 lazy import/动态加载可能
影响 eager wall time，故需记录独立进程的首次/再次 import 时间和正式计时稳定性。
先试直接复用持久 venv，若导入成本抵消收益，再把依赖 wheelhouse 持久保存并
在本地容器盘新建 venv；不要直接复制含绝对路径的 venv 到另一位置。
[Python venv 文档](https://docs.python.org/3/library/venv.html)说明了不可移植性。
