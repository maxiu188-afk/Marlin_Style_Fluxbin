# M=1 加速：本地准备与首次服务器执行顺序

2026-09-14：按最新用户指令，先参考 Marlin 做 M=1 kernel，再验证单个完整
transformer block，之后拓展全模型。未来接入 vLLM，本轮只保留接口。
本轮不连接服务器，不启动 GPU 实验，不构建或发布镜像，不新增精度实验。

当前交付是待 GPU 验证的实现与运行入口，不是已验收的 CUDA kernel 或加速结果。
本地 macOS/arm64 的 CPU 测试不能代替 NVIDIA 编译、正确性及计时验收。
固定输入仍是 `qwen3-8b-hybrid-distilled-step400-v1`，不重新量化。

## Marlin 参考与首版实现选择

只读参考本地 Marlin checkout，revision
`1f25790bdd49fba53106164a24666dade68d7c90`：

- `marlin/__init__.py` 的 `Layer.pack`：算法权重与执行布局分离、离线排列、
  调用方持有工作区。
- `marlin/marlin_cuda_kernel.cu`：紧凑权重读取、寄存器内解码、激活复用、
  沿 K 分工及归并；原实现还有 `cp.async` 多级流水、MMA 和跨 CTA 条带调度。
- 原 Marlin 为 FP16×INT4；本项目为两个 global bases 加两个 sparse bases、
  FP32 独立 row/column scales、g128/s8。直接调用 Marlin pack 会改变表示。

首版 `src/fluxbin_style/csrc/m1.cu` 是独立编写的 **SIMT packed GEMV**：
每 CTA 四个输出行，每行一 warp；每个 lane 读取连续的一个 packed byte，
解码四个输入位置；shared memory 复用输入、column scales 和选列映射。
默认每个 split 处理八组（K=1024），第二个 kernel 按固定顺序合并 FP32 部分和。
尾部输出行和不足八组的最后一个 split 有边界处理。

首版尚未实现 Marlin 的 tensor-core MMA、cp.async 多级流水或条带锁归并。
它是可测的正确性/性能起点；SIMT 是否合适、是否需要改变 CTA 宽度、split-K、
异步流水或 MMA 路径，要由 M=1 实测决定，不能从 Marlin 的 INT4 结果推断。

## 执行合同与布局 v1

格式 ID：`fluxbin-hybrid-g128-s8-m1-v1`。记 O 为输出维度，G=K/128。

| 字段 | 执行布局 | dtype |
|---|---|---|
| global signs | `[G,O,32]`，每 byte 四个位置、每位置两个 sign bits | uint8 |
| global row scales | `[G,O,2]` | FP32 |
| global column scales | `[G,128,2]` | FP32 |
| sparse signs | `[G,O,2]`，每组八个位置 | uint8 |
| sparse row scales | `[G,O,2]` | FP32 |
| sparse column scales | `[G,8,2]` | FP32 |
| 原选列索引 | `[G,8]` | int16 |
| 派生 lookup | `[G,128]`，未选为 -1，其余为 0..7 | int16 |

`convert_artifact` 仅排列、复制并生成 lookup；不重拟合、不改变 scales、符号、
选列。索引转 int16 在 0..127 范围内无损。lookup 额外占 `256*G` bytes，需计入
执行格式占用；不能把目录大小或算法 bpw 当作 kernel 格式占用。
`restore_artifact` 用于逐字段往返核对，`conversion_record` 记录源/目标 tensor hash。
运行时 layout 必须由已验证转换器创建，禁止直接传入未经检查的 lookup。

运算次序冻结为：

```text
term0 = FP32(sign0 * row0) * column0
term1 = FP32(sign1 * row1) * column1
global = FP32(term0 + term1)
selected positions: add separately reconstructed sparse pair in FP32
weight = cast(combined global + sparse, activation dtype)
dot = FP32 accumulation of input * weight
output = cast(dot, activation dtype)
```

必须对合并后的权重先舍入；把 row scales 移到点积外或分别计算 global/sparse
两个 GEMV 后相加，会改变目前 dense BF16 oracle 的语义，不属于无损布局优化。
编译禁用 FP32 乘加自动收缩（`--fmad=false`）；点积显式使用 FP32 FMA。

- 核心接口 `m1_out(x, layout, out, workspace)`：连续 `[1,K]`，BF16 主路径，
  FP16 辅助路径，同设备、当前 CUDA stream，SM80 或更新架构。
- CUDA v1 范围：`1<=O<=65536`、`K` 为 128 的倍数且 `G<=1024`。
- 核心算子不做分配、不隐式回退；扩展在 capture 之前显式编译/warmup。
- 工作区 `[ceil(G/groups_per_split),O]`、FP32，每次调用完全覆写，无需清零。
  不允许与输入/输出别名。并发 stream 必须由调用方提供独立工作区。
- `PackedHybridLinear` 是推理 adapter，默认拒绝非 M=1；仅显式开启
  `fallback='dense'` 才允许预填充等路径。该路径临时重建 dense 权重，不能声称
  prefill 加速或完整模型加载/显存已优化。
- 移动 adapter 只用 `.to(device)`；不能 `.half()`/`.bfloat16()` 改掉 FP32 scales。
- adapter 持有单个可复用工作区，适用于同 stream 顺序模型调用；未来 vLLM 的
  并发管理使用核心 ABI 自行管理工作区。

## 验收与阶段边界

1. 本地：字段往返、独立 materialization、异常输入、显式 fallback、block 替换
   覆盖与 engine contract 测试。
2. 首次 NVIDIA：编译及合成 CUDA 测试；包含 O 尾部、K split 尾部、FP16/BF16、
   非默认 stream、重复执行、NaN 污染工作区后覆写、CUDA Graph。
3. M=1 Linear：首先测 step400 的 layer 0 七个 Linears，覆盖四种真实形状
   `[4096,4096]`、`[1024,4096]`、`[12288,4096]` 以及 `[4096,12288]`。
   每个 module 单独记录；通过后可显式指定其他层，不能从 layer 0 声称全模型已验收。
4. 单个完整 block：`run_m1_block_probe.py` 只在 matching Linear trial 完成且稳定后
   显式启动；比较 packed 与同一 step400 的 dense BF16 block，另计原 BF16 block。
   首轮范围为真实权重、合成 hidden input、单 token、**空 KV cache**。包含
   RoPE、norm、attention、残差、七个 Linear；不代表有 KV 前缀的长上下文 decode。
5. 全模型：已准备 `replace_model_linears`，对 36 层/252 Linear 做覆盖检查并加载。
   实际完整模型 runner、固定 prompts、KV cache 长度、prefill/TTFT/decode 分离、
   token/logprob 门槛、每层 route 统计仍需在 block GPU 结果验收后准备并冻结。
   不能用这个安装接口的存在声称已经完成全模型 benchmark。
6. vLLM：仅预留 `engine_interface.LinearBackend` / `M1Backend` 的
   `convert / workspace_shape / apply_out` 接口和能力描述。没有导入 vLLM、注册量化
   方法、修改 scheduler/attention/KV cache、实现 fused QKV、TP sharding 或服务启动。
   未来另行固定 vLLM 版本并完成适配；接口不是 vLLM 兼容性验收。

数值 gate：BF16 的 `t=0.02`、FP16 的 `t=0.002`；要求输出有限、
`abs(error_i) <= t * (RMS(reference) + abs(reference_i)) + 1e-6`，且
`RMSE(error)/max(RMS(reference),1e-12) <= t/4`。重复执行需逐元素完全一致。
这些是首版工程容差，失败后诊断，不得为通过测试临时放宽。

计时：CUDA events，默认 warmup 20、每轮 100 次、7 轮交替 baseline/candidate 顺序；
跨轮 `(max-min)/median <= 10%` 才输出 Linear speedup。eager 和 graph 单独运行。
这是“device event 批量调用耗时”，不代表 host/API latency 或完整模型 wall time。
Linear 同轮 baseline 是 **解码后的 step400 dense BF16/FP16**，不是原 BF16 参数；
形状/dtype/输入一致。转换、JIT、模型加载不计入 operator 计时，另行记录。
block probe 有额外原 BF16 baseline。没有实现其他 low-bit backend 对比。

## 首次启动服务器：先记录，再安装，再验收

本轮不执行以下服务器命令。未来获得新 SSH 地址后，先核验原网络卷和 pinned
snapshot，不连接旧地址。环境记录以真实服务器为准，不能直接使用本机记录。
以下命令在仓库根目录、选定 Python 环境内执行，路径由环境变量传入。

1. **安装前**保存原模板 digest、Python/torch/CUDA、GPU/driver、nvcc、C++、ninja、
   已安装包版本、系统包、Git/source hashes、空闲磁盘。记录开始时间和所装工具。

```bash
python scripts/record_acceleration_environment.py \
  --phase before --image-reference "$FLUXBIN_BASE_IMAGE_DIGEST" \
  --output-dir "$FLUXBIN_RUN_ROOT/env-before"
```

`before` 在缺包时允许完成并写 `not_gpu_ready`。模板必须包含 CUDA **devel** 工具链
或补齐 nvcc；只有 torch 能访问 GPU 不意味着能编译扩展。

2. 保留模板已有 torch/CUDA/numpy，仅补缺少的编译工具、ninja 和模型依赖。
   使用该机器独立的环境；不要复制本机 venv。现有
   `requirements-linear-a100-v1.lock` 是历史模型依赖参考，先核对新模板再应用。
   `datasets` 并非 kernel 推理必需；不要为加速实验重新下载校准数据。
   先 `python -m pip install --no-deps -e .` 建立源码链接。

3. **安装后**记录并编译/执行合成 smoke。首次 compile 使用一个新的扩展缓存目录，
   记录 cold compile 耗时；之后复用按环境 ABI/GPU arch 隔离的持久缓存。
   可以使用现有 `infra/runpod/entrypoint.sh` 设置缓存变量，但不自动 Git pull。

```bash
python scripts/record_acceleration_environment.py \
  --phase after --image-reference "$FLUXBIN_BASE_IMAGE_DIGEST" \
  --require-cuda --build-smoke --output-dir "$FLUXBIN_RUN_ROOT/env-after"
```

记录器不会输出全部环境变量、token、pip 私有索引配置或带凭据的 freeze URL。
`packages.txt` 只是名称/版本清单。环境 JSON 保存环境/编译与测试状态，首次缺哪些
依赖由 before/after 差异确定，不提前猜成“已验证镜像”。

4. 环境通过后显式启动 **一项** Linear trial，不自动启动 block/full-model：

```bash
python scripts/run_m1_linear_benchmark.py \
  --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --environment "$FLUXBIN_RUN_ROOT/env-after/environment.json" \
  --output "$FLUXBIN_RUN_ROOT/linear-layer0-eager.json"
```

另一次 graph trial 使用 `--mode graph` 和新的 output 文件。
若要做内存安全检查，可在同样命令前加 `compute-sanitizer --tool memcheck`；
该运行只验正确性，不能用其计时作为性能结果。

5. Linear 人工审核后，另行启动单 block 探针：

```bash
python scripts/run_m1_block_probe.py \
  --snapshot-root "$FLUXBIN_SNAPSHOT_ROOT" --artifact-root "$FLUXBIN_ARTIFACT_ROOT" \
  --linear-result "$FLUXBIN_RUN_ROOT/linear-layer0-eager.json" \
  --environment "$FLUXBIN_RUN_ROOT/env-after/environment.json" \
  --output "$FLUXBIN_RUN_ROOT/block0-empty-cache.json"
```

正式 GPU 工作使用 tmux/调度器等持久机制启动；只确认启动并交接日志/任务 ID，
没有用户持续监控指令时不反复等待或自动推进下一阶段。

## 首次环境成功后固化镜像

`prepare_acceleration_image.py` 根据 before/after 生成构建目录；要求相同不可变
基础镜像 digest、通过 NVIDIA build smoke、完整系统包记录，且模板 torch/triton/
numpy 未被偷偷更换。输出实际新增/变更的 Python 包锁和带系统包版本的 Dockerfile。

```bash
python scripts/prepare_acceleration_image.py \
  --before "$FLUXBIN_RUN_ROOT/env-before/environment.json" \
  --after "$FLUXBIN_RUN_ROOT/env-after/environment.json" \
  --output-dir "$FLUXBIN_RUN_ROOT/image-bundle"
```

此命令不构建、不推送。之后在空间足够的 builder 上构建 linux/amd64 镜像，记录
最终 registry digest；不用此前因磁盘不足失败的 GitHub Actions 自动重试。
重新创建一个 Pod，挂载原卷、仅恢复 editable 源码链接与缓存变量，再用
`--phase recreated --require-cuda --build-smoke` 复验。比较首次/后续启动到可执行的
准备耗时后才接受镜像。驱动由 GPU host 提供，不打入镜像。

模型、数据、结果、密钥、持续变动源码不进镜像；稳定环境进镜像，模型/编译缓存
留网络卷，源码仍以 Git 同步。Kernel 源码变更会重新编译，不能把旧缓存当作验证。

## 本地检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

本地验证结果：75 项测试中 74 项通过，1 项 CUDA 测试因无 NVIDIA GPU 跳过；
包括缩小的真实 Qwen3 block API 的 CPU 对照。compileall 与 git diff --check 通过。

截至本轮，GPU 编译/真实权重 CUDA 数值/Linear 与 block 计时/全模型/vLLM/
镜像构建及重建验证均待后续服务器执行。准确结果仍以 `RESULTS_OVERVIEW.md` 为准。
