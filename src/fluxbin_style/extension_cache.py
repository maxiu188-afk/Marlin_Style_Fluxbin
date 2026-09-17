"""Content-addressed build directories for repository CUDA extensions.

The cache key deliberately excludes the Git revision. Documentation, runner,
or acceptance-gate changes must not force an identical CUDA translation unit to
compile again. Source bytes, flags, runtime ABI and target architecture remain
in the key, so an actual build-contract change always selects a new directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch


CACHE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_architecture() -> str:
    declared = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if declared:
        return declared
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return "unresolved"
    return f"{major}.{minor}"


@lru_cache(maxsize=8)
def _tool_identity(command: tuple[str, ...]) -> dict[str, object]:
    """Capture the executable and version without making the cache host-specific."""
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": list(command), "executable": "unresolved", "version": "unresolved"}
    try:
        completed = subprocess.run(
            [executable, *command[1:], "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        version = type(exc).__name__
    return {"command": list(command), "executable": str(Path(executable).resolve()), "version": version}


def _compiler_contract() -> dict[str, object]:
    cxx = tuple(shlex.split(os.environ.get("CXX", "c++")))
    if not cxx:
        cxx = ("c++",)
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = Path(cuda_home) / "bin/nvcc" if cuda_home else None
    nvcc_command = (str(nvcc),) if nvcc is not None and nvcc.is_file() else ("nvcc",)
    return {"cxx": _tool_identity(cxx), "nvcc": _tool_identity(nvcc_command)}


def extension_build_contract(
    *,
    name: str,
    sources: Iterable[Path | str],
    extra_cuda_cflags: Iterable[str],
) -> dict:
    """Return canonical inputs that may affect a compiled CUDA module."""
    source_paths = [Path(path).resolve() for path in sources]
    if not source_paths or any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("all CUDA extension sources must exist")
    if not name or "/" in name:
        raise ValueError("invalid extension name")
    contract = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "name": name,
        "sources": [
            {"name": path.name, "sha256": _sha256(path)} for path in source_paths
        ],
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "target_architecture": _target_architecture(),
        "cuda_home": os.environ.get("CUDA_HOME", "unresolved"),
        "compiler": _compiler_contract(),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "cache_key": hashlib.sha256(encoded).hexdigest()}


def content_addressed_build_directory(contract: dict) -> Path | None:
    """Create a persistent build directory when a cache root is configured."""
    root = os.environ.get("FLUXBIN_EXTENSION_CACHE_ROOT")
    if not root:
        root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        return None
    key = contract.get("cache_key")
    name = contract.get("name")
    if not isinstance(key, str) or len(key) != 64 or not isinstance(name, str):
        raise ValueError("invalid extension build contract")
    directory = Path(root).expanduser().resolve() / "content-v1" / key / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def extension_load_kwargs(
    *,
    name: str,
    sources: Iterable[Path | str],
    extra_cuda_cflags: Iterable[str],
    content_dependencies: Iterable[Path | str] = (),
) -> tuple[dict, dict]:
    """Return cpp_extension.load kwargs plus their auditable cache contract."""
    source_paths = [Path(path).resolve() for path in sources]
    dependency_paths = [Path(path).resolve() for path in content_dependencies]
    flags = list(extra_cuda_cflags)
    contract = extension_build_contract(
        name=name,
        sources=[*source_paths, *dependency_paths],
        extra_cuda_cflags=flags,
    )
    contract["compile_sources"] = [path.name for path in source_paths]
    contract["content_dependencies"] = [path.name for path in dependency_paths]
    kwargs = {
        "name": name,
        "sources": [str(path) for path in source_paths],
        "extra_cuda_cflags": flags,
        "verbose": True,
    }
    build_directory = content_addressed_build_directory(contract)
    if build_directory is not None:
        kwargs["build_directory"] = str(build_directory)
    return kwargs, contract
