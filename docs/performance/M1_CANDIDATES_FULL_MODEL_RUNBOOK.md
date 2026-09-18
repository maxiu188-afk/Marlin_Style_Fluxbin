# M=1 候选与完整模型运行手册

更新：2026-09-15。默认完整模型入口为下方 prepared v2.1；最新 Graph 两组均稳定，
相对原始 BF16 加速 1.381x/1.383x，eager 不稳定。详见
[加速结果](QWEN3_8B_M1_LINEAR_RESULTS.md)和[有效合同](ACCELERATION_HANDOFF.md)。
本页后半保留旧候选、动态 KV v1 和历史命令供复现，不表示应再次顺序执行全部实验。
vLLM 尚未接入；诊断、A8 和镜像自动化不因阅读本手册自动恢复。

## 当前完整模型：prepared v2.1（SXM4 已验证）

新入口 `scripts/run_qwen3_8b_prepared_m1_trial.py` 使用独立的
`configs/acceleration/qwen3_8b_full_m1_v2.json`；旧 v1 runner、配置和历史结果保留。
本轮直接测完整模型，不以单 block 的速度作为启动条件。

- 固定 `v5_p1024/gps1`，保持原来的 Qwen3-8B、step400 权重、两个真实 prompt、
  seed、batch=1、32 个相同 continuation token；主基线仍为原始 BF16。
- packed Linear 在预填充后绑定固定参数、输出和 workspace，计时内不重新构造布局字典、
  分配 packed 输出或更新 Python 审计计数。保留输入与 native 参数检查。
  输出复用仅支持串行单请求；绑定期间权重不可变，退出上下文或移动模块会解除绑定。
- 使用真实 prompt 预填充 StaticCache。每次测量前恢复 prefix KV 和长度；
  32 步中位置和有效注意力范围逐步增长。不是固定位置反复测一个 token，
  也不是随机初始化 KV。固定 token 的 Graph 不代表自由生成或服务吞吐。
- 比较 `prepared_eager` 和 `sequence_graph`：后者一次 replay 整段 32 步，包含
  embedding、全部 36 层、输出头和每步 argmax，以及 packed 的 LUT 构建与归约。
  reset、prefill、capture、校验和 CPU 输出拷贝均在 decode 计时外。
- 三个模型及两组 prompt cache 同时驻留，轮换 arm 顺序并交替两种模式。
  记录总驻留/峰值与真实执行顺序；这是新协议的显存条件，与 v1 的逐个加载不同。
  每种模式/arm/prompt 预热完整序列 8 次，测量 10 次；wall 和 CUDA event
  的 `(max-min)/median` 都必须 <=5% 才发布速度比。event 时间仍可能包含 host 发射间隙。
- 审计独立执行：同一静态 KV 下，原 checked wrapper 与 prepared wrapper 必须精确一致；Graph 与 prepared eager、
  每轮重复输出要求精确一致。检查 KV 长度、有限值、相同 fed tokens，以及全部 252 个
  packed Linear 各执行 32 次、decode dense fallback 为零。Graph coverage 记录 capture
  时的 Python 路由，replay 正确性另由完整输出检查，不将 capture 计数误称 replay 计数。
- packed 与 decoded BF16 的跨权重语义差异、动态/静态注意力路径差异均 report-only；不得将其称为数值等价。
  原始 BF16 为性能基线，decoded 为辅助基线。未接入 A8 或 vLLM。

服务器按下方“上机前与恢复”章节设置路径变量并生成**新的**环境记录后：

```bash
python -m unittest discover -s tests -p test_static_decode.py -v
python scripts/run_qwen3_8b_prepared_m1_trial.py \
  --snapshot-root "$FLUXBIN_SNAPSHOT_ROOT" \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output "$FLUXBIN_RUN_ROOT/full-model-prepared-v2.json"
```

使用项目已有的持久任务方式提交上述命令。GPU 测试必须执行而非 skip；若 Graph capture、
同路径数值或路由失败，runner 写入 failed 并停止，不静默降级。CPU 测试只验证小型模型的
缓存/上下文/恢复语义，不能证明 CUDA 可捕获或性能提升。v1 到 v2 同时改变了包装路径、
KV 与测量协议，因此不能把差值全部归因于某一项优化；历史 v1 负面结果保持不变，
v2.1 Graph 的稳定加速单独报告。

本地验证（macOS arm64，2026-09-15）：`python -m unittest discover -s tests`
共 113 项，105 通过、8 项 CUDA 测试跳过；新入口 `--help` 与 `git diff --check` 通过。
新增 CUDA 测试覆盖 prepared buffer 复用、输入改变后的 Graph replay，以及小型 packed
Qwen3 全序列的 KV/输出生命周期；这些测试随后已在本轮 SXM4 通过。

2026-09-15 SXM4 首次 v2 尝试：113/113 CUDA 环境测试通过；完整模型在 decoded
BF16 的动态/静态比较处停止（NRMSE 0.0121237，max logprob 0.491539，fed tokens
一致但预测不同），未开始计时。原始 failed JSON 保留。协议 v2.1 将动态/静态注意力
差异独立报告，并新增同一 StaticCache/mask 下 checked wrapper 的实测参照：prepared
包装必须与该参照逐位一致，Graph/repeat 同样逐位一致。此变更不放宽 wrapper 或
Graph 的正确性门槛，也不将动态/静态输出宣称等价；后续完整模型实测已完成，结论见下段。

本轮 v2.1 已完成：SXM4 Graph 对原始 BF16 为 1.381x/1.383x，两组稳定；eager
不稳定，整体 `completed_unstable`。113/113 GPU 测试通过，专项 7/7 复验通过。
完整证据与数值边界见 `QWEN3_8B_M1_LINEAR_RESULTS.md` 最新节。
新运行必须使用新输出目录，避免覆盖本轮证据。

## 历史候选与公共环境恢复

以下候选结果已归档；恢复环境命令仍适用，但每次必须指定新的结果目录。

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

## 历史诊断准备（用户已取消，不自动执行）

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

## 持久环境复用

2026-09-15 已实测：本次卷的持久 venv 导入约 30 秒，本地仅约 3 秒；
默认改用本地 venv，持久保存依赖和编译缓存。详见 [环境结果](../operations/RUNPOD_M1_SETUP_RESULTS.md)。
下述为可选持久 venv 入口，不能保证比本地重建更快。

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


v4 全模型启动策略补充：若 block 数值、重复、路径、来源检查均通过，仅计时
不稳定，用户已要求仍跑完整模型，可显式追加 `--allow-unstable-block-timing`。
原 block JSON/status/hash 不改，完整模型本身的 timing gate 不放宽。
默认不传参数仍要求稳定 block；该选项不允许数值失败或缺失/非有限样本。


## v5 LUT-A16 本地准备（2026-09-15）

使用 `configs/acceleration/m1_lut_candidates_v1.json`：v4/gps4 对照，v5 的
`groups_per_split=1/2/4/8/32` 五个候选。固定 BF16、layer 0 七项、eager/Graph、
原始 warmup/repeats/rounds 不变。建表和 reduction 均计入 packed 时间。
此次只做本地实现，没有启动 GPU 实验；以下在服务器已激活项目环境后执行。

先重新记录环境并通过 `--build-smoke`（现在包含 v5 CUDA 测试）。新增 LUT header
也在 source hash 中，旧环境记录不可直接复用。上文环境变量定义仍适用。

```bash
python scripts/run_m1_candidate_batch.py \
  --config configs/acceleration/m1_lut_candidates_v1.json \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output-dir "$FLUXBIN_RUN_ROOT/lut-candidates"
python scripts/summarize_m1_candidates.py \
  --config configs/acceleration/m1_lut_candidates_v1.json \
  --batch-dir "$FLUXBIN_RUN_ROOT/lut-candidates" \
  --output-dir "$FLUXBIN_RUN_ROOT/lut-summary"
```

审阅数值和稳定性后，将合格 v5 Linear JSON 设置为 `FLUXBIN_SELECTED_LINEAR_JSON`，
使用上文 block 命令。block 和 full-model 自动继承 v5/split，无需修改代码。
按用户已有要求，无论 block 是否加速都运行完整模型；若只是 block timing
不稳定，使用已有显式 override。数值/路由/来源失败不得用该开关跳过。
完整模型仍用原始 BF16 比速度，旧 decoded BF16 数值差异 report-only：

```bash
python scripts/run_qwen3_8b_full_m1_trial.py \
  --snapshot-root "$FLUXBIN_SNAPSHOT_ROOT" --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --block-result "$FLUXBIN_RUN_ROOT/block.json" \
  --numerical-policy report-only --allow-unstable-block-timing \
  --output "$FLUXBIN_RUN_ROOT/full-model-lut-reportonly.json"
```

服务器执行使用持久会话，保留日志、退出码和原始 JSON；本地测试通过不等于
CUDA 正确性或性能验收。vLLM 接口可选择 v5，但不连接 serving 框架。


## Inner compute 增量候选（2026-09-15）

新配置 `configs/acceleration/m1_inner_compute_candidates_v1.json`，12 个配置、
eager/Graph 共 24 trials。保留 v3/gps4、v4/gps4、v5/gps1；新增 v4_late 的
GPS 4/16/32，以及 v5_p256/p512/p1024 的 GPS 1/4。GPS 表示每个 K split 的组数。
其余输入、精度和计时协议不变。每 trial 上限 300 秒，整批最坏超过两小时；
正常实际耗时以日志为准，使用持久会话，不沿用旧整批 3700 秒 timeout。

按上文重新生成 source-bound 环境记录并通过 CUDA smoke，然后运行：

```bash
python scripts/run_m1_candidate_batch.py \
  --config configs/acceleration/m1_inner_compute_candidates_v1.json \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_NEW_ENVIRONMENT_JSON" \
  --output-dir "$FLUXBIN_RUN_ROOT/inner-candidates"
python scripts/summarize_m1_candidates.py \
  --config configs/acceleration/m1_inner_compute_candidates_v1.json \
  --batch-dir "$FLUXBIN_RUN_ROOT/inner-candidates" \
  --output-dir "$FLUXBIN_RUN_ROOT/inner-summary"
```

选定通过数值与稳定性审阅的 Linear JSON 后，block/full-model 自动继承对应
kernel 与 split，沿用上文完整模型 report-only 命令。不以 block 速度决定是否
提交完整模型。新增候选仍未在服务器编译或测量。
