# RunPod reusable environment

This setup minimizes GPU-Pod rebuild work while keeping accepted experiments
reproducible. It separates immutable software, persistent data, and frequently
changing source code.

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
