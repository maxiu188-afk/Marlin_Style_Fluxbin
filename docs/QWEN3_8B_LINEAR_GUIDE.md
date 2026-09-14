# Qwen3-8B：先验收 Linear，再扩展全模型

服务器环境更新（2026-09-13）：A100-SXM4-80GB 已连接。按用户要求，环境位于
container disk 的 `/opt/fluxbin-venv`，通过 `--system-site-packages` 复用模板的
PyTorch 2.8.0+cu128 / NumPy 2.1.2；Python 3.12.3。新增 Transformers 5.14.1、
datasets 5.0.0、safetensors 0.8.0 及其依赖，环境约 410 MB。`pip check` 和随机初始化
小型 Qwen3 的 CUDA/BF16 前向测试通过。新增依赖清单见
`infra/runpod/requirements-linear-a100-v1.lock`，服务器完整版本记录在
`/opt/fluxbin-environment-freeze.txt`。模型和实验数据仍规划放 `/workspace`；
随后已完成真实 8B 下载、预检及五个代表性 Linear 验收，结果见
[Linear 验收表](QWEN3_8B_LINEAR_RESULTS.md)。全模型 pure/hybrid 的产物完整性与权重重建已带备注验收，
见[全模型验收报告](QWEN3_8B_FULL_RESULTS.md)。匹配 PPL 执行已验收，
但两组量化方案均未通过质量门槛，见[PPL 验收报告](QWEN3_8B_PPL_RESULTS.md)。

服务器上运行下列 Python 命令前，用 `source /opt/fluxbin-venv/bin/activate`
激活环境。Container disk 是临时环境，Pod 重建后需恢复；它不承担结果持久化。

2026-09-13：第一部分代码及真实 Linear 验收均已完成；以下保留可复现的分步操作。
这里的“精度”指权重重构及校准输入下的 Linear 输出误差；任务准确率和全模型 PPL 尚未覆盖。

## 当前实现

沿用 32B 的 `hessian_obq.py`、`two_base_rank1.py` 和 packing 算法，历史入口不变。
新增独立的 8B 入口，避免旧 GH200/32B 常量被误用于新模型：

| 文件 | 作用 |
| --- | --- |
| `src/fluxbin_style/qwen3_8b.py` | 8B 架构、清单、离线快照检查及 Linear 结果判定 |
| `scripts/prepare_qwen3_8b_linear.py` | 扫描本地快照，生成已绑定权重哈希的目标配置 |
| `scripts/materialize_qwen3_8b_c4_calibration.py` | 使用 8B tokenizer 生成独立 C4 校准产物 |
| `scripts/run_qwen3_8b_single_linear_hessian_obq_s8.py` | 一个目标的 Hessian 捕获、pure/hybrid OBQ、指标和产物检查 |
| `configs/experiments/qwen3_8b_single_linear_hessian_obq_s8_v1.json` | 待快照预检解析的模板，不可直接运行真实量化 |
| `configs/calibration/qwen3_8b_c4_256x2048_v1.json` | 新校准协议，不复用 32B token 哈希 |

第一轮只处理第 0 层。一次调用只验证一个目标，不启动其他目标或全模型。

| `--target` | `[O,K]` | 用途 |
| --- | --- | --- |
| `self_attn.o_proj` | `[4096,4096]` | 首个参考 Linear |
| `self_attn.q_proj` | `[4096,4096]` | 不同输入位置的 attention 投影 |
| `self_attn.k_proj` | `[1024,4096]` | 非方形 attention 投影 |
| `mlp.gate_proj` | `[12288,4096]` | MLP 扩张 |
| `mlp.down_proj` | `[4096,12288]` | MLP 收缩、最大 Hessian 维度 |

所有目标由完整 BF16 模型的第 0 层捕获输入，并各自从同一原始权重/Hessian 开始，
pure/hybrid 不共享拟合结果。不是先量化 q_proj 再捕获下游输入；后者属于全模型阶段。

## 不变的算法约定

- 两个二值基、独立 row/column scales、group size 128、精确四种 sign assignment。
- 原 greedy 初始化，ALS 最多 50 次，自适应 assignment 内存预算 1 GiB。
- C4 256×2048、seed `20260902`，原数据集 revision、采样方法保持不变。
- `H = 2 X^T X / activation_rows`，1% mean-diagonal damping、Cholesky inversion。
- Hybrid 每组按 Hessian saliency 选 8 列，保留稳定 tie-breaking 和独立 OBQ 传播。

## 服务器开好后的操作顺序

先使用 RunPod 现成 PyTorch/CUDA 模板；检查实际 GPU、Python、PyTorch、Transformers、
datasets 和 safetensors，再补缺失或经确认不兼容的包。不要直接重装整套 PyTorch/CUDA。
历史依赖范围并不证明任意模板都兼容；本地测试也不替代模板上的 smoke check。

以下命令是未来执行指引，本次没有执行。进入仓库根目录，使用该服务器独立 Python 环境。
先准备本地存在的固定版本快照，设置 `FLUXBIN_SNAPSHOT` 为其目录；目录名须为
`b968826d9c46dd6066d109eabc6255188de91218`。这些 runner 不下载模型。

### 1. 离线预检并准备 o_proj 配置

```bash
PYTHONPATH=src python scripts/prepare_qwen3_8b_linear.py \
  --snapshot-root "${FLUXBIN_SNAPSHOT:?请先设置本地快照目录}" \
  --target self_attn.o_proj \
  --output results/qwen3-8b-linear-v1/o_proj.config.json
```

检查全部 shard 哈希、索引与实际 Safetensors 清单、36 层/252 Linear/6,945,767,424 权重、
BF16 dtype、目标形状、tokenizer 文件哈希。快照目录名本身不是来源证明；下载时还需记录
固定 revision 的获取记录。本入口把实际字节哈希固定下来供后续复查，不能伪称已经远程验真。

### 2. 单独生成一次 C4 校准数据

```bash
PYTHONPATH=src python scripts/materialize_qwen3_8b_c4_calibration.py \
  --config configs/calibration/qwen3_8b_c4_256x2048_v1.json \
  --tokenizer-root "${FLUXBIN_SNAPSHOT:?请先设置本地快照目录}" \
  --tokens artifacts/qwen3-8b-c4-v1/tokens.safetensors \
  --output artifacts/qwen3-8b-c4-v1/manifest.json \
  --source-manifest artifacts/qwen3-8b-c4-v1/source.sha256
```

这一步可能下载指定的一个 C4 shard，需要事先确认网络和空间。保持 32B 的采样规则，
但使用 8B tokenizer 重新生成，不将 32B 产物改名冒充 8B。已存在文件会拒绝覆盖。

### 3. CUDA 上只运行 o_proj

```bash
PYTHONPATH=src python scripts/run_qwen3_8b_single_linear_hessian_obq_s8.py \
  --config results/qwen3-8b-linear-v1/o_proj.config.json \
  --snapshot-root "${FLUXBIN_SNAPSHOT:?请先设置本地快照目录}" \
  --calibration-manifest artifacts/qwen3-8b-c4-v1/manifest.json \
  --calibration-tokens artifacts/qwen3-8b-c4-v1/tokens.safetensors \
  --payload artifacts/qwen3-8b-linear-v1/o_proj.safetensors \
  --output results/qwen3-8b-linear-v1/o_proj.json \
  --source-manifest results/qwen3-8b-linear-v1/o_proj.source.json
```

先人工验收结果，然后依次换 target 和各输出文件名运行其余四个目标，复用同一校准产物。
命令为前台入口；真实长任务应使用 Pod 上已验证的 tmux/持久会话并交接日志，
本轮不额外安装工具、不自动提交任务。

## Linear 验收看什么

- 权重 SSE、MSE、relative Frobenius，以及同一 Hessian 下的校准输出误差必须有限。
- 明确要求 hybrid 的权重 SSE 和校准输出损失均严格低于 pure。
  任一项未改善，保留诊断产物、标记 `failed_linear_gate`，进程返回 2，不掩盖失败。
- 非选中列 delta 必须精确为零；索引必须组内有效、排序且唯一。
- sign pack/unpack 必须精确一致；写入后重新打开，检查 10 个 payload tensor 的哈希和清单。
- 通过自动检查时仍标记 `completed_pending_review`，不得把执行完成当作人工验收。
- 记录实际 GPU、compute capability、PyTorch/CUDA/Transformers 版本、峰值显存和耗时。
  初次运行用于环境和精度验证，不能当作 packed 推理加速证据。

## 下一部分与当前边界

五个目标全部验收后，才移植独立 pure/hybrid 的 36 层全模型传播、逐层恢复和完整产物验收。
之后才是匹配 BF16/pure/hybrid 的 WikiText-2 PPL。新增的全模型入口为
`scripts/run_qwen3_8b_full_hessian_obq_s8.py`，配置模板为
`configs/experiments/qwen3_8b_full_hessian_obq_s8_v1.json`。使用
`scripts/prepare_qwen3_8b_full.py --project-root . --output <新配置路径> --suite-output <新门槛路径>`
绑定五个目标的 acceptance/result/payload/config 哈希后，才可执行。
未实现新的 8B PPL runner；5% PPL 部署门槛继续按 `EXPERIMENT_PLAN.md` 执行。

全模型入口保留历史逐层算法、独立分支和恢复语义，固定当前 A100 软件版本；
`--validate-only` 仅验收输入，`--stop-after-layer` 保留有界回放功能。
必须使用 `artifacts/qwen3-8b-full-hessian-obq-s8-v1/pure` 和
`artifacts/qwen3-8b-full-hessian-obq-s8-v1/hybrid_s8` 分别保存权重。
实际提交前检查全部五个 Linear 的验收，不因入口已写好而越过门槛。

本地已验证：全部 45 项测试通过，覆盖 Linear 合成验证、全模型边界和恢复一致性、
五目标验收完整性、坏指标与文件篡改拒绝。真实 GPU Linear 结果以服务器结果和独立
acceptance JSON 为准；它们不是全模型质量或 PPL 证据。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```
