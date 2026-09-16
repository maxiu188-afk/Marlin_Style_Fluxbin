# GPTQ W3 corrected routes: A100 full-model runbook

更新：2026-09-17。本页只用于在 A100 80GB 上比较两个已经通过 RTX PRO 4500
Linear 诊断的 decoded-BF16 修正路线。今天只完成接入与 CPU 静态验证；完整模型
correctness 和性能仍是 pending，不能从 Linear 结果外推。

## 冻结候选

两条路线使用相同的 W3 payload、row tile、prompt、seed、32-token StaticCache decode、
prepared wrapper 和整段 CUDA Graph 协议。差别只有 `groups_per_split`：

| route | q/o `[4096,4096]` | k/v `[1024,4096]` | gate/up `[12288,4096]` | down `[4096,12288]` |
|---|---:|---:|---:|---:|
| `fast_corrected` | 1 | 1 | 1 | 1 |
| `observed_exact` | 1 | 1 | 4 | 2 |

两条路线的 arithmetic 均为 `decoded_bf16`。`fast_corrected` 是 Linear 级速度优先候选；
`observed_exact` 是 RTX 固定 layer-0/input 上四类 shape 均逐位匹配 decoded oracle 的
观测候选。“observed exact”不代表完整模型或所有输入已证明 bit-exact。

Linear 诊断结果 SHA256 为
`4f567a9adf28d80eb2f7d08f08146a08a855d80f3c9589f02d5bd845f51460ca`；配置把该哈希、
A100 compute capability 8.0 和至少 75,000,000,000 bytes VRAM 作为启动门。

## 固定入口

- runner：`scripts/run_qwen3_8b_w3_full_m1_trial.py`
- protocol：`configs/acceleration/qwen3_8b_w3_corrected_full_m1_v1.json`
- adapter：`src/fluxbin_style/w3_lut_deployment.py`
- artifact loader：`src/fluxbin_style/w3_lut_artifacts.py`

旧 `qwen3_8b_w3_full_m1_v1.json` 仍默认走 structural/gps1，没有被改写为新结果。

## 2026-09-18 A100 运行

使用保留网络卷中的 snapshot 和 W3 layout。先确认仓库干净并 fast-forward 到本次提交：

```bash
export project_root=/workspace/repos/marlin-style-fluxbin
export snapshot_root=/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export w3_layout_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1
export job_root=/workspace/jobs/qwen3-8b-w3-corrected-full-m1-v1-a100-20260918
export result_root=/workspace/results/qwen3-8b-w3-corrected-full-m1-v1-a100-20260918
cd "$project_root"
git status --short --branch
git pull --ff-only
git rev-parse HEAD
nvidia-smi
python -m pip check
```

确认设备后做专项测试并记录与当前源码匹配的环境：

```bash
python - <<'PY'
import torch
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory = torch.cuda.get_device_properties(0).total_memory
print(name, capability, memory)
assert "NVIDIA A100" in name
assert capability == (8, 0)
assert memory >= 75_000_000_000
PY
python -m unittest discover -s tests -p 'test_w3*.py' -v
mkdir -p "$job_root" "$result_root"
export image_reference='<provider-image-tag-or-digest>'
python scripts/record_acceleration_environment.py \
  --output-dir "$result_root/environment" \
  --phase recreated \
  --image-reference "$image_reference" \
  --require-cuda
export w3_layout_manifest_sha256="$(sha256sum "$w3_layout_root/manifest.json" | cut -d' ' -f1)"
test "$w3_layout_manifest_sha256" = f2825dda33d77491364fc857f3c4e36be114be859cf26aa76d142e0e2441edf8
```

用独立 tmux 启动。首次返回只检查任务已脱离前台并开始写日志，不持续轮询：

```bash
tmux new-session -d -s w3-corrected-full \
  "bash -lc 'cd \"$project_root\" && set -o pipefail; python scripts/run_qwen3_8b_w3_full_m1_trial.py --protocol configs/acceleration/qwen3_8b_w3_corrected_full_m1_v1.json --snapshot-root \"$snapshot_root\" --w3-layout-root \"$w3_layout_root\" --w3-layout-manifest-sha256 \"$w3_layout_manifest_sha256\" --environment \"$result_root/environment/environment.json\" --output \"$result_root/result.json\" 2>&1 | tee \"$job_root/run.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/exit-code\"'"
tmux has-session -t w3-corrected-full
tail -n 30 "$job_root/run.log"
```

取消命令：

```bash
tmux kill-session -t w3-corrected-full
```

## 验收

作业结束后检查退出码、结构化结果和哈希：

```bash
cat "$job_root/exit-code"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["result_root"]) / "result.json"
report = json.loads(path.read_text())
print(report["status"])
for arm, gate in report["candidate_acceptance"].items():
    print(arm, gate)
assert set(report["candidate_acceptance"]) == {
    "packed_w3_fast_corrected",
    "packed_w3_observed_exact",
}
assert report["all_dynamic_static_checks_passed"] is True
assert all(
    prompt["prepared_wrapper_exact"] and prompt["graph_exact"]
    for arm in report["arms"].values()
    for prompt in arm["prompts"]
)
PY
sha256sum "$result_root/result.json" "$result_root/environment/environment.json"
```

每条 candidate 独立判定：两个 prompt 的 primary `sequence_graph` 必须同时通过
packed-vs-decoded W3 数值门（logits NRMSE 0.005、max log-prob 0.05、greedy/fed
tokens 相同）和 5% timing-range 门，才可标为 `accepted=true`。速度分别对同轮
original BF16 和 decoded W3 报告；不得用一条路线的通过替代另一条。
