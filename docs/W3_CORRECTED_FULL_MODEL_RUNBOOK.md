# GPTQ W3 corrected routes: A100 full-model runbook

更新：2026-09-17。本页记录两个 decoded-BF16 修正路线在 A100 80GB 上的完成结果，
并保留可复现命令。正式作业已退出 0；两条路线都获得加速，但都没有通过完整模型
correctness。性能数字按 raw CUDA median 报告，因为全比较 stability gate 被 isolated
outlier 触发，不能从 Linear 结果或中位数改写为 formal acceptance。

## 实际执行结果

- 设备：NVIDIA A100 80GB PCIe，compute capability 8.0。
- 源码 revision：`d0f85b49745d169b6ffc1f4a7e4d97a197e6911d`。
- 正式 retry1：2026-09-17 02:25:55Z--03:00:14Z，退出 0，状态
  `completed_with_backend_numerical_differences`。
- `fast_corrected`：354.399 / 355.055 ms，相对 original BF16 为
  **1.344074x / 1.342683x**。
- `observed_exact`：374.968 / 375.391 ms，相对 original BF16 为
  **1.270342x / 1.269945x**。
- 两条 candidate 的 `accepted` 都是 false。fast logits NRMSE 为
  0.020764 / 0.022001；observed 为 0.019733 / 0.017023，均超过 0.005。
- prepared wrapper、Graph 和 252 条 packed route 检查通过，dense fallback 为 0。
- formal timing gate 因 device 与 wall-time 的 isolated outlier 未全通过；fast candidate
  自身两组 CUDA device 计时稳定，但 wall-time 有 host-side outlier。去掉 min/max 后，
  八个 CUDA device timing cell 的相对极差均不超过 0.26%。

按性能优先口径，两条 corrected 路线均不如历史 structural W3 的约 1.49x；当前
性能主线仍是 structural，corrected 结果作为数值语义和额外开销的负面 follow-up。

首个 tmux launch 因 quoting 选到 system Python，在模型加载前失败。retry1 显式使用
venv Python 后完成；因此下面的复现模板要求先固定并验证 `python_bin`。

结果与 provenance：

- 远端：`/workspace/results/qwen3-8b-w3-corrected-full-m1-v1-a100-pcie-20260917-retry1/`。
- 本地私有备份：
  `server_results/runpod_w3_corrected_full_m1_a100_pcie_2026-09-17/`。
- result SHA256：`08271b47c7db3e5197557a6fef25af659cf90e885621e7d4660a99d3c3a2c0cd`。
- environment SHA256：`9f045ce580131aac8f717d53ea7fe1a85c5c8f389c5fce0111ce30afed23091e`。
- protocol SHA256：`f7d3f12cc29a645e1d9862c4fc245a8ffc32b22428136e6b8c86427b55256b24`。
- runner SHA256：`511a9ac78e3fc5437ed19833fe4b71f47c2899798503197871e0dd163547054e`。

最后审计时 GPU、CPU 实验进程和 tmux 均为空；实例 shutdown-ready，但实际电源状态
仍由用户确认。

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

## A100 复现模板

使用保留网络卷中的 snapshot 和 W3 layout。先确认仓库干净并 fast-forward 到本次提交：

```bash
export project_root=/workspace/repos/marlin-style-fluxbin
export snapshot_root=/workspace/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export w3_layout_root=/workspace/models/fluxbin/qwen3-8b-gptq-w3-lut-planar-v1
export job_root=/workspace/jobs/<new-w3-corrected-run-id>
export result_root=/workspace/results/<new-w3-corrected-run-id>
export python_bin=/absolute/path/to/project-venv/bin/python
cd "$project_root"
git status --short --branch
git pull --ff-only
git rev-parse HEAD
nvidia-smi
"$python_bin" -m pip check
"$python_bin" -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

确认设备后做专项测试并记录与当前源码匹配的环境：

```bash
"$python_bin" - <<'PY'
import torch
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory = torch.cuda.get_device_properties(0).total_memory
print(name, capability, memory)
assert "NVIDIA A100" in name
assert capability == (8, 0)
assert memory >= 75_000_000_000
PY
"$python_bin" -m unittest discover -s tests -p 'test_w3*.py' -v
mkdir -p "$job_root" "$result_root"
export image_reference='<provider-image-tag-or-digest>'
"$python_bin" scripts/record_acceleration_environment.py \
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
  "bash -lc 'cd \"$project_root\" && set -o pipefail; \"$python_bin\" scripts/run_qwen3_8b_w3_full_m1_trial.py --protocol configs/acceleration/qwen3_8b_w3_corrected_full_m1_v1.json --snapshot-root \"$snapshot_root\" --w3-layout-root \"$w3_layout_root\" --w3-layout-manifest-sha256 \"$w3_layout_manifest_sha256\" --environment \"$result_root/environment/environment.json\" --output \"$result_root/result.json\" 2>&1 | tee \"$job_root/run.log\"; printf \"%s\\n\" \"\${PIPESTATUS[0]}\" > \"$job_root/exit-code\"'"
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
"$python_bin" - <<'PY'
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
assert all(
    prompt["prepared_wrapper_exact"] and prompt["graph_exact"]
    for arm in report["arms"].values()
    for prompt in arm["prompts"]
)
print("all_dynamic_static_checks_passed", report["all_dynamic_static_checks_passed"])
print("all_timings_stable", report["all_timings_stable"])
print("primary_timings_stable", report["primary_timings_stable"])
PY
sha256sum "$result_root/result.json" "$result_root/environment/environment.json"
```

每条 candidate 独立判定：两个 prompt 的 primary `sequence_graph` 必须同时通过
packed-vs-decoded W3 数值门（logits NRMSE 0.005、max log-prob 0.05、greedy/fed
tokens 相同）和 5% timing-range 门，才可标为 `accepted=true`。速度分别对同轮
original BF16 和 decoded W3 报告；不得用一条路线的通过替代另一条。
`all_dynamic_static_checks_passed` 是独立 report-only 字段，不应被误写成 candidate
acceptance 的硬断言。本次它和全部 candidate acceptance 都为 false，但
prepared-wrapper/Graph exact 与 route coverage 仍单独通过。
