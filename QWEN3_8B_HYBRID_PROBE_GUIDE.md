# Hybrid 补偿修复：开机前准备与首轮验证

2026-09-13：本地准备完成，54 项测试通过；尚未运行 CUDA 或真实模型对照。
只研究 hybrid，蒸馏留到诊断之后。旧量化器、旧权重和 16.142104 PPL 记录保持不变。

## 本轮唯一改动

新增 `src/fluxbin_style/hybrid_conditioned.py`。对完整逆 Hessian 的上三角
Cholesky 因子 U，使用 `solve_triangular(U_BB, U_BR)` 计算当前剩余子问题
的补偿系数，避免重复使用未经条件化的原始逆矩阵切片。多 group 测试用直接
求逆的剩余 Hessian 和自由坐标梯度为独立 oracle；identity 情形精确匹配旧实现。

真实对照包含两组：

1. `legacy_hybrid`：用当前旧算法重新拟合，保存选列结果。
2. `conditioned_fixed_indices`：修复补偿，但完全复用第一组列索引。

第一组是同输入的新基准，不是直接拿旧全模型 payload 作比较。两组的后续工作
权重会因补偿而不同，这是本轮研究的变化；显著列索引、初始化方法、组大小、
拟合预算和输入保持一致。选列策略更新、增加迭代、层内顺序校准均不在本轮改动中。

## 首批真实目标和指标

- `model.layers.1.mlp.gate_proj`：[12288,4096]。
- `model.layers.1.mlp.up_proj`：[12288,4096]，与上项共用输入 Hessian。
- `model.layers.6.mlp.down_proj`：[4096,12288]。
- 输入统一来自原始 BF16 模型前缀，采用已有 C4 256×2048 tokens。
  这是控制变量的 Linear 诊断，不能声称重现旧 hybrid 量化前缀或全模型 PPL。
- 固定 g=128、s=8、50 次 ALS 上限、1% damping、原始列选择公式；禁用 TF32。
- 分别存储并重新读取 payload，解码成 BF16 后计分；检查列索引完全一致。
- 主指标：直接计算每个 token 的 `||X(W−Q)^T||²` 后取均值（不除输出维度）。
  同时与 `0.5 * trace((W−Q)H(W−Q)^T)` 比较，误差容限 1e-4 relative / 1e-6 absolute。
- 第二次原始 BF16 前缀遍历直接计算输出误差；两次每批输入的有序张量哈希必须相同。
- 记录权重 SSE、主输出误差、payload / W / H / 输入 / 源码哈希、拟合时间和显存。
  局部时间不是性能 benchmark，不据此宣称内核加速。
- 运行完成只生成 `completed_pending_review`；误差未改善也保留结果，不能强行判成功。
  不自动提交全模型、PPL、pure 或蒸馏。

冻结配置：`configs/experiments/qwen3_8b_hybrid_compensation_probe_v1.json`。
本轮运行器：`scripts/run_qwen3_8b_hybrid_compensation_probe.py`。

## 启动服务器之后

复用保留的存储和已验证的 A100-SXM4-80GB / PyTorch 2.8.0+cu128 模板。
如选择不同 GPU 或软件版本，先检查兼容性并建立新配置，不绕过运行时检查。
原始模型、C4 和旧 hybrid 权重无需重下。以下命令在服务器执行，从 checkout 根目录开始。

1. 确认保留存储已挂载，Git 工作区干净，再通过 GitHub 同步 `codex/runpod-bootstrap`。
   不通过文件复制同步源码。新 SSH 地址交给 Codex 后可完成此步骤。
2. `/opt` 环境若已丢失，先检查模板的 Python、torch、CUDA；在 container disk 重建
   `/opt/fluxbin-venv`（`--system-site-packages`），仅按
   `infra/runpod/requirements-linear-a100-v1.lock` 恢复附加依赖。不要复制本机 venv。
3. 激活环境后先跑测试与预检；以下保留存储路径对应上一台 Pod 的布局。

```bash
source /opt/fluxbin-venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests
```

只验证输入与运行时，不拟合：

```bash
python scripts/run_qwen3_8b_hybrid_compensation_probe.py \
  --snapshot-root /workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --calibration-dir "$PWD/artifacts/qwen3-8b-c4-v1" \
  --output-dir /workspace/jobs/hybrid-compensation-probe-v1/artifacts
```

正式运行（仅在前两步通过后执行）：

```bash
mkdir -p /workspace/jobs
tmux new-session -d -s hybrid-compensation-probe-v1 \
  "bash scripts/run_hybrid_probe_job.sh /workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 '$PWD/artifacts/qwen3-8b-c4-v1' /workspace/jobs/hybrid-compensation-probe-v1"
```

任务包装要求全新 job 目录，重复执行不会覆盖旧日志；如失败，先诊断再使用新的
明确命名目录。tmux 与子进程会在 SSH 断开后继续运行。源码哈希写入 provenance
并在结束时核对，运行中不要更新代码。

只读查看命令：

```bash
tail -n 30 /workspace/jobs/hybrid-compensation-probe-v1/probe.log
```

结束后查看 `exit-code`；结果在 job 目录下 `artifacts/result.json`，中间拟合记录
在 `artifacts/fits.json`，冻结来源在 `artifacts/provenance.json`。未生成结果不等于通过。
需要取消时，先读取 `probe.pid` 并核实进程，再单独执行 `kill <已核实的PID>`；
取消命令不放入上述可整块复制的操作中。

## 本地验证范围

54 项测试包括历史回归、相关 Hessian 的多 group 条件最优性、旧切片反例、
identity 与旧 hybrid 一致、固定索引/输入不变、稀疏支持、payload BF16 解码一致、
前缀提前终止和输入重放。CLI 帮助及 Bash 语法检查通过。
尚未验证：真实 A100 上的 Cholesky 数值稳定性、同输入直接误差与 Hessian 误差的
容限、真实拟合效果及资源消耗。开机后先运行上述有界对照，结果不预设改善。
