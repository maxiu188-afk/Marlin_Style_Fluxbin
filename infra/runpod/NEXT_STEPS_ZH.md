# Docker 安装后的操作指引

> 2026-09-13：用户已改为先使用 RunPod 现成 PyTorch/CUDA 模板，按实际缺项补依赖。
> 以下自定义镜像流程暂缓，不再是开服务器或验证 Linear 的前提。
> 当前操作入口请看 [8B Linear 指引](../../QWEN3_8B_LINEAR_GUIDE.md)。

更新：2026-09-12。以下是操作指引，不代表其中的构建、发布或实验已经执行。

## 现在到哪里了

| 项目 | 当前状态 |
| --- | --- |
| Docker 客户端与引擎 | 已验证可用，版本 29.7.2 |
| 本机 | macOS / Apple M5；Docker VM 约 8 GB 内存 |
| Buildx | `desktop-linux` 正常，支持 `linux/amd64` |
| 项目镜像 | Dockerfile、依赖锁定文件和本地构建脚本已准备；尚未构建验收 |
| 镜像发布 | 尚无已验收 registry digest |
| RunPod | 按交接记录，尚未创建 Volume、Template 或 Pod |
| Qwen3-8B | 尚未开始真实模型量化或 PPL |

当前最近的一步是：**本机构建镜像 → 本地依赖检查 → 发布镜像并记录 digest**。
完成后才配置 RunPod，最后进入 8B 的 A0 预检。

命令按步骤分块执行，不要一次复制全文。下面 macOS 命令均从本仓库根目录执行。
文中没有夹带取消、删除或清理命令。

## 1. 准备当前终端

本次安装没有创建系统级 Docker 命令链接。使用 Docker.app 自带命令，
只对当前终端设置 PATH，不需要修改全局配置。

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
export DOCKER_CONTEXT=desktop-linux
export BUILDX_BUILDER=desktop-linux
docker info --format 'Server={{.ServerVersion}} OS={{.OSType}} Arch={{.Architecture}}'
docker buildx inspect desktop-linux
```

预期：Server 返回版本，builder 状态正常且包含 `linux/amd64`。
本机 engine 显示 `aarch64` 正常，构建目标仍明确指定为 `linux/amd64`。
另一个 `default` context 因缺少系统 socket 报错时，不要误判为 Desktop 引擎失败。

在 Docker Desktop 的资源设置中确认虚拟磁盘的容量和剩余空间；Mac 的空闲磁盘
不等于 Docker VM 的可用空间。先保留当前约 8 GB VM 内存，若出现资源错误再定位调整。

只读查看占用：

```bash
df -h .
docker system df
```

## 2. 构建项目镜像

```bash
bash infra/runpod/build-local.sh
```

这会使用现有 RunPod CUDA/PyTorch 基础镜像安装项目依赖，构建 `linux/amd64`，
并将结果载入本地 Docker。不会上传模型、创建 Pod 或启动实验。

本条是前台构建：保持终端和 Docker Desktop 运行，避免 Mac 休眠。
若需要关闭终端，让 Codex 另行以持久后台任务启动并交接 PID 和日志，不能直接关闭前台任务。
首次构建耗时取决于基础镜像下载、解压、依赖安装及跨架构模拟，不能用 Docker 安装耗时估算。

构建完成条件：脚本退出码为 0，打印 `Built ...`，并生成以下文件：

- `tmp/runpod-build/build.log`：构建日志。
- `tmp/runpod-build/metadata.json`：Buildx 元数据。
- `tmp/runpod-build/image-inspect.json`：本地镜像信息。

需要查看进度时，在另一个位于仓库根目录的终端执行这一条只读命令：

```bash
tail -n 40 tmp/runpod-build/build.log
```

失败时先看最后一条明确错误：连接重置可重试并复用缓存；磁盘不足先确认 Docker VM
空间；依赖版本不存在或冲突则核对 `requirements.lock`，不要擅自升级为 latest。
本地构建元数据不等于已发布的 registry digest。

### Docker Hub token 请求出现 connection reset

2026-09-12 的排查中，直接访问 `auth.docker.io` 出现连接重置，而经本机已有
HTTP 代理 `127.0.0.1:10808` 请求同一 token 接口返回 HTTP 200。
这是当时本机的代理地址，不是其他机器通用的配置。
在已完成第 1 步 PATH 设置的终端中，为 Docker CLI 的认证请求设置代理后重试：

```bash
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
bash infra/runpod/build-local.sh
```

以上只影响当前终端及其子进程。如果仍在拉取镜像层时失败，检查 Docker Desktop
Settings 中的 Proxies 是否使用系统代理。CLI 环境变量和 Desktop 后端代理是不同的
配置层；不要把 `daemon.json` 的代理设置当作 Docker Desktop 的配置方法。
不要因错误中包含 IPv6 地址就直接关闭 IPv6，也无需删除 Dockerfile 的 syntax 行。

## 3. 检查镜像架构和 Python 环境

以下操作需要第 2 步成功，并沿用第 1 步的终端环境。

```bash
docker image inspect marlin-style-fluxbin:cu1281-torch280-v1 \
  --format '{{.Os}}/{{.Architecture}} {{.Id}}'
docker run --rm --platform linux/amd64 \
  marlin-style-fluxbin:cu1281-torch280-v1 python -m pip check
docker run --rm --platform linux/amd64 \
  marlin-style-fluxbin:cu1281-torch280-v1 \
  python -c 'import torch, transformers, datasets, safetensors; print("torch", torch.__version__); print("CUDA runtime", torch.version.cuda); print("transformers", transformers.__version__); print("datasets", datasets.__version__); print("GPU available", torch.cuda.is_available())'
```

通过条件：镜像为 `linux/amd64`，`pip check` 无依赖冲突，关键模块可导入。
Mac 上 `GPU available=False` 是预期结果。这只能验收镜像的本地基础环境，
真实 CUDA 和性能必须在 NVIDIA 主机上验证。记录实际版本，与 Dockerfile/lock 对照。

## 4. 发布到 GHCR，并冻结镜像 digest

这是上传镜像的步骤，第 3 步通过后再执行。只发布运行环境，不把权重、结果、
缓存或凭据加进构建上下文；现有 `.dockerignore` 排除了这些项目目录。

先在终端自行完成 GHCR 登录。用户名使用 `maxiu188-afk`；凭据按 GitHub 官方说明
使用有相应 package 权限的 token，在交互提示中输入，不写进文档或命令参数。

```bash
docker login ghcr.io -u maxiu188-afk
```

确认本次版本标签尚未作为已验收镜像使用，然后发布：

```bash
docker tag marlin-style-fluxbin:cu1281-torch280-v1 \
  ghcr.io/maxiu188-afk/marlin-style-fluxbin:cu1281-torch280-v1
docker push ghcr.io/maxiu188-afk/marlin-style-fluxbin:cu1281-torch280-v1
docker buildx imagetools inspect \
  ghcr.io/maxiu188-afk/marlin-style-fluxbin:cu1281-torch280-v1
```

保存 push/inspect 返回的 `sha256:...`、源码提交、构建日志和依赖版本。
RunPod 应使用 `ghcr.io/maxiu188-afk/marlin-style-fluxbin@sha256:实际摘要`。
这个字符串是格式说明，必须换成真实摘要。以后修改环境使用新版本标签。
私有 GHCR 镜像还需要在 RunPod 配置只读拉取凭据。

## 5. 配置 RunPod 持久存储与 Template

此步骤会创建计费资源，应在镜像发布成功后由用户在控制台确认资源和费用。

1. 先确认目标数据中心的 GPU 供应；计划 H20 开发、H200 和 A100 80GB 最终评测。
2. 按项目方案准备约 100 GB Network Volume 起步，挂载 `/workspace`。
   容量是否足够须根据模型、双分支产物和缓存实测，不把 100 GB 当作保证。
3. 创建 Template，使用上一步真实 image digest；容器磁盘按现有方案 25–30 GB 起步，
   并核对实际镜像要求。保留镜像入口与默认 `sleep infinity`。
4. 设置 `PERSIST_ROOT=/workspace`、`FLUXBIN_ENV_ID=cu1281-torch280-v1`。
   凭据通过平台 secret/credential 功能配置。
5. 创建 Pod 后先验证可进入终端。当前项目入口本身不启动 SSH 服务，
   不要把安装了 `openssh-client` 当成可 SSH 登录；如需 SSH，应单独配置并验证服务。

持久目录：模型/数据/结果放 `/workspace`；环境放镜像；代码通过 Git 同步。
Network Volume 可在 Pod 终止后保留，但仍产生存储费用，并受数据中心位置约束。
不要把普通 Pod volume 和 Network Volume 混为一谈。

## 6. 在 NVIDIA Pod 内验收环境

以下命令仅在 Pod 内执行，不在 Mac 执行：

```bash
nvidia-smi
python -m pip check
python - <<'PY'
import platform
import torch
import transformers
print('Python', platform.python_version())
print('PyTorch', torch.__version__)
print('CUDA runtime', torch.version.cuda)
print('Transformers', transformers.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU', torch.cuda.get_device_name(0))
print('Capability', torch.cuda.get_device_capability(0))
x = torch.ones(16, device='cuda')
assert x.sum().item() == 16
print('CUDA smoke check passed')
PY
```

记录 image digest、GPU/driver、软件版本、输出路径、剩余空间。
上述检查是环境 smoke check，不是实验数值或速度验收。

之后将已审查的代码通过 Git 同步到 `/workspace/repos`，在仓库内执行
`python -m pip install --no-deps -e .`。正式运行必须固定源码提交并记录工作区状态。
当前本地构建脚本/文档尚未提交，推送前只纳入本次预期文件，保留已有未跟踪分析脚本。

## 7. 环境就绪后才开始 8B A0

按 [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md) 执行：

1. 创建新的 8B 配置和 runner，保留历史 32B 配置与产物。
2. 核对固定模型 revision、tokenizer、36 层 / 252 个 Linear 及权重清单。
3. 固定 C4 256×2048、seed `20260902`、校准产物及哈希。
4. 通过本地合成检查和 NVIDIA 环境预检，再进入代表性真实 Linear。
5. 代表性 Linear 通过后完整量化；完整产物验收后进行匹配 BF16/pure/hybrid PPL。
6. 只有相对 BF16 PPL 差距不超过 5% 的候选进入 packed 部署。

现有 runner 仍有 32B 假设，因此现在没有可以直接复制执行的已验收 8B 全量命令。
各阶段单独验收，不自动连跑。详细状态见 [CURRENT_HANDOFF.md](../../CURRENT_HANDOFF.md)。

## 官方参考

- [Docker 跨架构构建](https://docs.docker.com/build/building/multi-platform/)：Apple Silicon 上的模拟构建可能明显慢于原生构建。
- [GitHub GHCR 登录与发布](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)。
- [RunPod 存储类型](https://docs.runpod.io/pods/storage/types)：创建资源时再次核对当前规则及费用。
