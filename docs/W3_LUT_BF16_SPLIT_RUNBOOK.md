# GPTQ W3 BF16 舍入修正与 split-G 诊断手册

更新：2026-09-16。本轮是独立的诊断实验，不覆盖既有 4×3 Linear 结果，也不自动
进入完整模型。目标只有两个：验证 `±3` BF16 权重舍入修正是否消除 retained
decoded-BF16 差异，以及测量 `groups_per_split`、single-split 直接写回和 R2048 的
Linear 级空间。

## 实验边界

- 当前 structural arithmetic 继续作为对照，`decoded_bf16` 是新修正候选。
- 新候选不增加 payload；它从三个已有 bit-plane 推导 `q=+3/-3` mask，并增加两次
  activation-LUT lookup 来修正 `BF16(3s)-3s`。
- 四类真实 shape 仍固定使用 Qwen3-8B layer 0，输入 seed、BF16、group 128、关闭
  TF32/reduced-precision reduction，与第一轮保持一致。
- 46 个 cell 均使用同轮 CUDA Graph total；另外分别计时 gps1 的 main 和 finish，
  该分阶段计时只用于诊断，不能相加替代 Graph total。
- 输出状态为 diagnostic-only。Linear 通过不能自动更新完整模型 correctness 或性能结论。

## 固定入口

- 配置：`configs/acceleration/w3_lut_bf16_split_candidates_v1.json`
- runner：`scripts/run_w3_lut_bf16_split_trial.py`
- kernel：`src/fluxbin_style/csrc/w3_lut.cu`
- CPU/静态测试：`tests/test_gptq_w3_planar.py`、
  `tests/test_w3_lut_bf16_split_trial.py`、`tests/test_w3_lut_correctness.py`

## 服务器恢复与预检

复用保留的网络卷和既有模型，不覆盖旧结果目录：

```bash
export project_root=/workspace/repos/marlin-style-fluxbin
export gptq_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-g128-sym-v1
export w3_layout_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1
export job_root=/workspace/jobs/qwen3-8b-w3-bf16-split-v1
export result_root=/workspace/results/qwen3-8b-w3-bf16-split-v1
mkdir -p "$job_root" "$result_root"
cd "$project_root"
git status --short --branch
git rev-parse HEAD
nvidia-smi
python -m pip check
python -m unittest discover -s tests -p 'test_w3*.py' -v
```

使用供应商页面显示的真实镜像引用，不得猜测：

```bash
export image_reference='<provider-image-tag-or-digest>'
python scripts/record_acceleration_environment.py \
  --output-dir "$result_root/environment" \
  --phase recreated \
  --image-reference "$image_reference" \
  --require-cuda
```

检查既有 layout，记录 manifest hash：

```bash
test -f "$gptq_root/decoded_bf16/layer-000.safetensors"
test -f "$w3_layout_root/manifest.json"
export w3_layout_manifest_sha256="$(sha256sum "$w3_layout_root/manifest.json" | cut -d' ' -f1)"
```

## 启动诊断 trial

任务用独立 tmux 运行；启动后只做一次有界检查：

```bash
tmux new-session -d -s w3-bf16-split \
  "bash -lc 'cd \"$project_root\" && set -o pipefail; python scripts/run_w3_lut_bf16_split_trial.py --environment \"$result_root/environment/environment.json\" --gptq-root \"$gptq_root\" --w3-layout-root \"$w3_layout_root\" --w3-layout-manifest-sha256 \"$w3_layout_manifest_sha256\" --output \"$result_root/result.json\" 2>&1 | tee \"$job_root/run.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/exit-code\"'"
tmux has-session -t w3-bf16-split
tail -n 30 "$job_root/run.log"
```

取消命令：

```bash
tmux kill-session -t w3-bf16-split
```

## 验收

```bash
cat "$job_root/exit-code"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["result_root"]) / "result.json"
report = json.loads(path.read_text())
print(report["status"], len(report["cells"]), len(report["stage_timings"]))
assert len(report["cells"]) == 46
assert len(report["stage_timings"]) == 8
assert report["diagnostic_only"] is True
for cell in report["cells"]:
    assert cell["correctness"]["passed"] is True
    assert cell["correctness"]["repeat_exact"] is True
PY
```

审阅顺序：

1. 比较每个 shape 的 `reference_checks.corrected_vs_decoded` 与
   `structural_vs_decoded`，确认修正是否真正降低误差。
2. 只在 correctness、repeat 和稳定性同时通过的 cell 中选择最小 latency。
3. 对比 gps1 main/finish 分阶段时间，判断 reduction/第二次 launch 的占比。
4. 只有四个 shape 都得到明确数值结论后，才设计完整模型候选；不得把本轮 Linear
   结果直接写成完整模型 accepted。
