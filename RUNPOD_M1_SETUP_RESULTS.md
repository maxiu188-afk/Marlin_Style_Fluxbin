# 首次 M=1 服务器环境验收

2026-09-14，结论：**环境和小规模合成 CUDA 正确性通过**。
状态 `accepted_environment_and_synthetic_cuda`。不是实模型 kernel 或性能验收。

| 项目 | 已核验结果 |
|---|---|
| GPU | NVIDIA A100 80GB PCIe，SM80 |
| 驱动 | 595.91.07 |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128，保留模板 torch |
| CUDA runtime / nvcc | 12.8 / 12.8.93；nvcc 原已安装，只需补 PATH |
| 主要 Python 包 | numpy 2.1.2、transformers 5.14.1、safetensors 0.8.0、datasets 5.0.0 |
| 新增系统工具 | ninja-build 1.11.1-2 |
| 编译合同 | SM80，--fmad=false，未启用 --use_fast_math |
| 首次编译及 smoke | 6 项通过，57.79 秒，包括进程启动及测试，并非纯编译耗时 |
| 全套服务器测试 | 75/75 通过，无跳过，7.882 秒 |
| 任务退出 / 依赖一致性 | 0 / pip check 通过 |
| 审核时源码 | `2868d5fa9763bfaa90ea1db786dea8a4c923783e`，工作区干净 |

审计核对了 source hashes、编译产物及 flags、日志和退出码、安装前后环境、
当前 GPU/torch 和依赖一致性；没有为本次验收重复运行全部测试。
CUDA 测试包含 FP16/BF16、输出尾部、split 尾部、非默认 stream、重复输出、
工作区污染后覆写及 CUDA Graph。测试仍为小矩阵合成输入，不涵盖真实 Qwen3 shapes。
缩小的 Qwen3 block API 对照是 CPU/reference 测试，不是 GPU block benchmark。

持久卷与 step400 的 43 个文件已在启动阶段逐文件核验。原 snapshot 目录仍在，
其 shards 本轮没有重新哈希；后续需要完整模型时按 frozen config 检查。

## 证据绑定

- 安装前环境 SHA256：`89813f02670908400833683102a9c8dd04132b8145671d1e4a9836924d8d052f`
- 安装后环境 SHA256：`3d292a30e057242042abbc53116b99d3b6e6ef68733c9536e8e8283a4976d977`
- setup.log SHA256：`54ae6d9220cdac30c7bb9dbcfc7ce2fe35c6e6414963fd2368d9933987cbe10e`
- CUDA .so SHA256：`28ed497aba407e90717471bff822ba4b8675dfeb98efc7b9af3232a553d05d2b`
- build.ninja SHA256：`ead9adc10fd53776bc37569b104598b267056c37b19a463f3c73a31a5ce5b62a`

远端结果与验收：`/workspace/results/m1-a100-20260914/`。
远端完整日志：`/workspace/jobs/m1-a100-20260914-setup-v1/setup.log`。
本地小型证据：`server_results/runpod_m1_a100_2026-09-14/`，Git-ignored。

## 后续入口与镜像边界

下次运行先激活已保存的 `runtime.sh`：其中包含独立 venv、CUDA_HOME/PATH、
TORCH_CUDA_ARCH_LIST、持久 extension cache 和 MAX_JOBS；不能假设普通 SSH shell
已继承安装任务的环境变量。

```bash
source /workspace/results/m1-a100-20260914/runtime.sh
cd /workspace/repos/marlin-style-fluxbin
```

下一阶段是固定 step400、layer 0 七个真实 Linear 的 M=1 数值与计时验收。
本次没有启动该阶段，没有 block/full-model/vLLM 或速度结论。

安装前后依赖和系统工具版本已保留。基础镜像仅有用户提供的“RunPod 默认模板”
描述，名称/digest 未确认；因此当前不能生成通过不可变基础镜像检查的构建材料，
更不能声称已验证复用镜像。后续补齐镜像身份，再构建、重建 Pod 并复验。
