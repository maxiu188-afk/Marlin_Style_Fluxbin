# GPTQ W3 inline LUT：服务器运行手册

更新：2026-09-16。本手册只覆盖第一轮 inline W3 LUT：先确认 GPTQModel 7.4.0
的 canonical `g_idx/desc_act` 语义，再离线规范化，最后完成四类真实 Qwen3-8B
Linear shape × 三个 row tile 的 12-cell trial。`prepare` 不在本轮实现或 benchmark。

## 冻结合同

- 输入：既有 `GPTQ W3 / group=128 / sym=True / desc_act=True` 原始 artifact。
- raw checkpoint 按 qzero format-v1 解码；不得把已加载的 GPTQ-v2 qzero 语义套到
  raw safetensors。
- 每个 Linear 独立计算 `stable_argsort(g_idx)`；weight 的 K 维和 runtime activation
  使用同一 permutation。
- converter 必须先完成全部 36 层、252 个 Linear 的 canonical-semantics 扫描，
  全部通过后才创建输出目录。
- 正式候选只有 `R=256/512/1024` 三个 inline 版本。
- 正式性能数字只有同一 job、同一输入、同一轮次的 CUDA Graph total；candidate
  graph 覆盖 inline LUT build、main 和 finish。JIT、load、离线转换不计时。
- 四类真实 shape 固定为 q `[4096,4096]`、k `[1024,4096]`、gate
  `[12288,4096]`、down `[4096,12288]`，均取 layer 0 的实际权重。
- 同轮基线固定为 decoded W3 BF16、原始 BF16 和 `v5_p1024/gps1`。

`prepare` 只允许在 12-cell 结果完成后，对每个 shape 的最佳稳定 inline 版本做
profile，并且 profiler 证明 LUT construction 是主要瓶颈时，另开诊断分支。

## 固定路径

挂载保留的 `/workspace` 网络卷后设置：

```bash
export project_root=/workspace/repos/marlin-style-fluxbin
export snapshot_root=/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export gptq_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1
export qbb_root=/workspace/models/fluxbin/qwen3-8b-hybrid-distilled-step400-v1
export w3_layout_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1
export job_root=/workspace/jobs/qwen3-8b-w3-lut-inline-v1
export result_root=/workspace/results/qwen3-8b-w3-lut-inline-v1
mkdir -p "$job_root" "$result_root"
cd "$project_root"
```

不要删除或覆盖已存在的 `w3_layout_root`、`job_root` 或 `result_root`。若路径已存在，
先检查 manifest、日志和 Git revision，再决定使用新的版本化目录。

## Stage 0：Git、硬件和静态检查

代码必须先通过 Git 正常同步到服务器；不要复制本机 venv。确认 checkout、数据和 GPU：

```bash
git status --short --branch
git rev-parse HEAD
nvidia-smi
test -f "$gptq_root/manifest.json"
test -f "$qbb_root/manifest.json"
test -d "$snapshot_root"
python -m pip check
python -m unittest discover -s tests -p 'test_gptq_w3_planar.py' -v
python -m unittest discover -s tests -p 'test_lut_primitives.py' -v
python -m unittest discover -s tests -p 'test_w3_lut_config.py' -v
```

用供应商页面显示的真实镜像 tag 或 digest 替换变量，不要猜测：

```bash
export image_reference='<provider-image-tag-or-digest>'
python scripts/record_acceleration_environment.py \
  --output-dir "$result_root/environment" \
  --phase after \
  --image-reference "$image_reference" \
  --require-cuda
```

环境记录必须为 `ready_for_gpu_trial`。随后编译三个 inline 版本，并在 synthetic
FP16/BF16 上检查 eager、重复执行和 CUDA Graph replay；这一步不计时：

```bash
python scripts/run_w3_lut_server_preflight.py \
  --environment "$result_root/environment/environment.json" \
  --output "$result_root/cuda-preflight.json" \
  2>&1 | tee "$job_root/cuda-preflight.log"
```

只有 `cuda-preflight.json` 为 `passed` 才进入真实 artifact。

## Stage 1：先确认 canonical semantics

这个命令逐个读取全部 252 个 raw Linear，但不创建 deployment payload：

```bash
tmux new-session -d -s w3lut-semantics \
  "bash -lc 'cd \"$project_root\" && set -o pipefail; python scripts/prepare_qwen3_8b_w3_lut_artifacts.py --gptq-root \"$gptq_root\" --validate-only --inspection-output \"$result_root/canonical-semantics.json\" 2>&1 | tee \"$job_root/canonical-semantics.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/canonical-semantics.exit\"'"
tmux has-session -t w3lut-semantics
tail -n 20 "$job_root/canonical-semantics.log"
```

验收 `status=canonical_semantics_confirmed`、`linears=252`、每组严格 128 列、decoded
zero 只有 4；`nonidentity_permutations` 只做事实记录，不要求为零。完成后
`canonical-semantics.exit` 必须为 0。

## Stage 2：离线规范化

转换会再次 fail-closed 检查每个模块，并逐项要求 integer code、decoded zero、raw
FP16 scale 来源，以及 GPTQModel loaded-BF16 scale cast 和 retained decoded-BF16
bitwise exact。它先写
`w3_layout_root.incomplete`，全部通过后才原子改名为正式目录。用 tmux 持久执行：

```bash
tmux new-session -d -s w3lut-convert \
  "bash -lc 'cd \"$project_root\" && set -o pipefail; python scripts/prepare_qwen3_8b_w3_lut_artifacts.py --gptq-root \"$gptq_root\" --output-dir \"$w3_layout_root\" 2>&1 | tee \"$job_root/conversion.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/conversion.exit\"'"
```

只做一次有界启动检查后交还监控：

```bash
tmux has-session -t w3lut-convert
tail -n 20 "$job_root/conversion.log"
```

完成后检查：

```bash
cat "$job_root/conversion.exit"
sha256sum "$w3_layout_root/manifest.json" | tee "$result_root/w3-layout-manifest.sha256"
```

退出码必须为 0，manifest 状态必须为 `completed_pending_gpu_validation`。若失败，
保留 `.incomplete/failure.json` 诊断，不要把 staging 目录改名成正式 artifact。

## Stage 3：正式 4×3 trial

把上一阶段输出的 manifest SHA256 作为显式参数；runner 会再次核对环境、源码、GPU、
artifact hash、真实 shape 和 12-cell 覆盖：

```bash
export w3_layout_manifest_sha256="$(cut -d' ' -f1 "$result_root/w3-layout-manifest.sha256")"
tmux new-session -d -s w3lut-4x3 \
  "bash -lc 'cd \"$project_root\" && set -o pipefail; python scripts/run_w3_lut_benchmark.py --environment \"$result_root/environment/environment.json\" --cuda-preflight \"$result_root/cuda-preflight.json\" --snapshot-root \"$snapshot_root\" --gptq-root \"$gptq_root\" --w3-layout-root \"$w3_layout_root\" --w3-layout-manifest-sha256 \"$w3_layout_manifest_sha256\" --qbb-artifact-root \"$qbb_root\" --output \"$result_root/w3-lut-inline-4x3.json\" 2>&1 | tee \"$job_root/benchmark.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/benchmark.exit\"'"
```

有界启动检查：

```bash
tmux has-session -t w3lut-4x3
tail -n 20 "$job_root/benchmark.log"
```

完成验收：

```bash
cat "$job_root/benchmark.exit"
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["result_root"]) / "w3-lut-inline-4x3.json"
r = json.loads(p.read_text())
print(r["status"], len(r["cells"]), r["metric"], r["prepare_in_round_one"])
assert len(r["cells"]) == 12
assert r["metric"] == "same-run CUDA Graph total per call"
assert r["prepare_in_round_one"] is False
PY
```

`completed_pending_review` 表示 12 格都通过正确性且各臂稳定；
`completed_with_unstable_cells` 只表示结果保留待分析，不得发布对应 speedup。

## 状态与取消

```bash
tmux ls
tail -n 40 "$job_root/conversion.log"
tail -n 40 "$job_root/benchmark.log"
tmux kill-session -t w3lut-convert
tmux kill-session -t w3lut-4x3
tmux kill-session -t w3lut-semantics
```

不要在本轮自动启动 profiler 或新增 prepare 实现。先审阅 12-cell JSON，选出每个 shape
的最佳稳定 inline candidate，再单独冻结 profiling 命令和触发判据。
