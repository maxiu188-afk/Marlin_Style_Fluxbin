# RunPod reusable environment

2026-09-14: first acceleration-server startup must capture the template environment
before installation and the working environment after CUDA build smoke. Use
`scripts/record_acceleration_environment.py`; generate a future image bundle with
`scripts/prepare_acceleration_image.py` only from verified observations. The full
procedure is in [M1 preparation](../../docs/performance/M1_ACCELERATION_PREPARATION.md). No server
connection or image build is authorized for the current offline preparation turn.
The older provisioning/image details below are historical or future guidance.

2026-09-13: custom image work is deferred. The current route uses an existing
RunPod PyTorch/CUDA template, inspects installed packages, and adds only needed
dependencies. See [the 8B Linear guide](../../docs/quality/QWEN3_8B_LINEAR_GUIDE.md).
The image lifecycle below remains a future reproducibility option.

This setup minimizes GPU-Pod rebuild work while keeping accepted experiments
reproducible. It separates immutable software, persistent data, and frequently
changing source code.

## Current execution status

RunPod provisioning is **paused** as of 2026-09-12.

- GitHub Actions run
  [`34681093330`](https://github.com/maxiu188-afk/Marlin_Style_Fluxbin/actions/runs/34681093330)
  started from commit `e041f77c8edf5c2ce095c18ec003e66582039e83`.
- Checkout, Buildx setup, and GHCR login succeeded.
- The `Build and push linux/amd64 image` step failed when the GitHub-hosted
  runner reported `No space left on device`.
- The digest-recording step did not run. No image tag/digest is accepted for an
  experiment, even if an incomplete registry upload is later visible.
- No RunPod Network Volume, Template, Pod, or registry credential was created.
- The workflow is manual-dispatch only. Do not rerun it until the build is
  changed to fit the runner disk or moved to a builder with sufficient space.

A future resume must first choose and document one build-space repair, such as
using a smaller pinned base, removing unnecessary hosted-runner toolchains
before Buildx, disabling the large `mode=max` build cache, or building on a
machine with sufficient disk. Then run the workflow manually, require a
successful digest, and only afterward configure RunPod storage and Templates.

### Local Docker preparation (2026-09-12)

The user authorized installing Docker and preparing the image locally. Docker
Desktop's client and engine are now verified at version 29.7.2 on the Apple
Silicon development Mac. The `desktop-linux` builder supports `linux/amd64`;
the CLI is available inside Docker.app without system-wide binary links. The local host
has approximately 331 GiB free, so the selected build-space strategy is a local
Buildx build, retaining the existing CUDA/PyTorch base and avoiding a `mode=max`
cache export. GitHub Actions remains manual and idle.

See [the Chinese next-steps guide](NEXT_STEPS_ZH.md) for terminal setup, local
build/validation, registry publication, RunPod setup, and the 8B A0 boundary.

After installation, launch Docker Desktop and complete its first-run setup.
Check its VM disk allocation and available space before building the large CUDA
image. From the repository root:

```bash
docker info
bash infra/runpod/build-local.sh
```

The helper explicitly targets `linux/amd64`, loads the image locally, and saves
build logs, metadata, and image inspection under Git-ignored `tmp/runpod-build/`.
If the CLI is not on PATH, set `DOCKER_BIN` to the installed Docker CLI path.
The helper does not publish an image or start experiments. A successful build,
registry publication with a recorded digest, and an NVIDIA-host preflight are
still required. No local image build has been started or accepted yet.

## Three layers

1. **Custom image**: CUDA/PyTorch, compiler toolchain, Python dependencies, and
   the cache-routing entrypoint. Do not put model weights, calibration data,
   results, credentials, or frequently changing repository code in the image.
2. **Network Volume at `/workspace`**: Hugging Face cache, datasets, accepted
   quantized artifacts, results, source checkouts, Torch extension cache, and
   Triton cache. This survives Pod termination and can be attached to a newly
   created Pod in the same supported data-center context.
3. **RunPod Template**: immutable image tag/digest, disk sizes, volume mount,
   ports, and non-secret environment configuration.

During development, keep the Git checkout under `/workspace/repos` and use an
editable install. Once a result becomes a formal gate, record the clean Git
revision and image digest. Bake a stable CUDA extension into a new image only
after its source/API stops changing; JIT build caches remain performance aids,
not provenance.

## Build and publish

RunPod uses `linux/amd64`. On an Apple Silicon Mac, build and push with Buildx:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file infra/runpod/Dockerfile \
  --tag ghcr.io/OWNER/marlin-style-fluxbin:cu1281-torch280-v1 \
  --push \
  .
```

Replace `OWNER`, use an immutable version tag rather than `latest`, and record
the registry digest returned after the push. If the registry is private, add
read-only registry credentials in RunPod; never place tokens in the Dockerfile,
image layers, repository, or Template environment text.

The provided base is a bootstrap environment, not yet the frozen formal
benchmark environment. Before the first accepted Qwen3-8B job:

- confirm the RunPod host driver supports the image CUDA version;
- run `nvidia-smi`, record the driver/GPU, and check `torch.cuda.is_available()`;
- record Python, PyTorch, CUDA, Transformers, Triton, compiler, and image digest;
- either accept this new pinned stack for the entire study or build a second
  image matching the historical software versions. Do not mix environments
  within a matched comparison.

## Recommended Template

- Image: the immutable registry tag above, preferably resolved and recorded by
  digest.
- Container disk: 25-30 GB; it is disposable and contains only the image and
  temporary files.
- Storage: a 100 GB Network Volume mounted at `/workspace` to start. Increase it
  only after measuring the Qwen3-8B snapshot and artifact sizes; RunPod volumes
  cannot be reduced.
- Command: keep the image default `sleep infinity` so SSH/tmux work remains
  available.
- GPU: select H20 for normal development; create separate Pods from the same
  Template for H200 and A100 80GB validation.
- Environment: `PERSIST_ROOT=/workspace` and an immutable
  `FLUXBIN_ENV_ID`. Change the environment ID when CUDA, PyTorch, Triton, or the
  extension ABI changes. The entrypoint automatically adds the detected
  `sm80`/`sm90` capability to compiled-cache paths; a Template may set
  `FLUXBIN_CACHE_ARCH` explicitly if detection is unavailable.
- Secrets: configure registry/Hugging Face tokens through RunPod secret or
  credential controls, not as image `ENV` values.

Network Volumes for Pods are currently a Secure Cloud feature and constrain
GPU selection to the volume's data center. Before allocating a large volume,
check that H20, H200, and A100 80GB are available in that location. If one data
center cannot supply all three, use one volume per required data center and
sync immutable artifacts through object storage; RunPod does not automatically
replicate Network Volumes.

## Persistent layout

The entrypoint creates:

```text
/workspace/
  cache/
    huggingface/
    pip/
    torch-extensions/<environment-id>-<gpu-arch>/
    triton/<environment-id>-<gpu-arch>/
  datasets/
  models/
  repos/
  artifacts/
  results/
```

Keep authoritative artifacts and results outside build caches. A cache may be
deleted and regenerated; an accepted payload/result must have a manifest and an
external backup.

Example first checkout:

```bash
cd /workspace/repos
git clone GIT_REPOSITORY_URL marlin-style-fluxbin
cd marlin-style-fluxbin
python -m pip install -e .
```

For subsequent Pods, the checkout is already on the Network Volume. The
editable-install marker lives in the disposable container, so recreate only
that cheap link with `--no-deps`; do not reinstall the dependency stack:

```bash
cd /workspace/repos/marlin-style-fluxbin
git status --short --branch
python -m pip install --no-deps -e .
python - <<'PY'
import platform
import torch
import transformers

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

Do not automatically pull or checkout a branch in the container entrypoint.
Experiments must not change revision merely because a Pod restarted.

## Stop versus terminate

- Container-disk data is disposable and is lost on stop/restart.
- A normal Pod volume at `/workspace` survives stop/start but is deleted when
  the Pod is terminated. It is tied to that Pod and stopped storage remains
  billable.
- A Network Volume survives Pod termination and can be attached to a new Pod.
  Pods with Network Volumes are terminated/recreated rather than stopped.

For this project, prefer **terminate compute + retain Network Volume + recreate
from Template**. It supports changing GPU type without rebuilding dependencies
or redownloading the model. Use ordinary stop/start only when staying on one Pod
and accepting its storage cost and possible restart-capacity limitations.

## Formal-run preflight

Before every accepted run, save a small provenance record containing:

- image repository, immutable tag, and digest;
- Git revision and clean/dirty state;
- GPU name, compute capability, driver, CUDA runtime/toolkit, PyTorch, Triton,
  Transformers, and compiler versions;
- hashes of model/config/token/calibration inputs;
- `FLUXBIN_ENV_ID` and the exact kernel build flags;
- writable output path and free-space check.

The image and Network Volume eliminate reconstruction time. They do not replace
the experiment's source, input, payload, and result hashes.
