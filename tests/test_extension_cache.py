import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fluxbin_style.extension_cache import (
    extension_build_contract,
    extension_load_kwargs,
)


class ExtensionCacheTests(unittest.TestCase):
    def test_key_tracks_content_flags_and_arch_not_git_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.cu"
            source.write_text("__global__ void k() {}\n")
            with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0"}, clear=False):
                first = extension_build_contract(
                    name="example", sources=[source], extra_cuda_cflags=["-O3"]
                )
                same = extension_build_contract(
                    name="example", sources=[source], extra_cuda_cflags=["-O3"]
                )
                changed_flags = extension_build_contract(
                    name="example", sources=[source], extra_cuda_cflags=["-O2"]
                )
                source.write_text("__global__ void k() { return; }\n")
                changed_source = extension_build_contract(
                    name="example", sources=[source], extra_cuda_cflags=["-O3"]
                )
            self.assertEqual(first["cache_key"], same["cache_key"])
            self.assertNotEqual(first["cache_key"], changed_flags["cache_key"])
            self.assertNotEqual(first["cache_key"], changed_source["cache_key"])
            self.assertNotIn("git", first)

    def test_explicit_persistent_root_selects_content_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.cu"
            source.write_text("__global__ void k() {}\n")
            with patch.dict(
                os.environ,
                {
                    "FLUXBIN_EXTENSION_CACHE_ROOT": str(root / "cache"),
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                },
                clear=False,
            ):
                kwargs, contract = extension_load_kwargs(
                    name="example", sources=[source], extra_cuda_cflags=["-O3"]
                )
            build = Path(kwargs["build_directory"])
            self.assertTrue(build.is_dir())
            self.assertEqual(build.name, "example")
            self.assertEqual(build.parent.name, contract["cache_key"])
            self.assertEqual(build.parent.parent.name, "content-v1")

    def test_content_dependency_invalidates_key_without_becoming_compile_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.cu"
            header = root / "kernel.cuh"
            source.write_text('#include "kernel.cuh"\n')
            header.write_text("constexpr int value = 1;\n")
            with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0"}, clear=False):
                first_kwargs, first = extension_load_kwargs(
                    name="example",
                    sources=[source],
                    content_dependencies=[header],
                    extra_cuda_cflags=["-O3"],
                )
                header.write_text("constexpr int value = 2;\n")
                second_kwargs, second = extension_load_kwargs(
                    name="example",
                    sources=[source],
                    content_dependencies=[header],
                    extra_cuda_cflags=["-O3"],
                )
            self.assertNotEqual(first["cache_key"], second["cache_key"])
            self.assertEqual(first_kwargs["sources"], [str(source.resolve())])
            self.assertEqual(second_kwargs["sources"], [str(source.resolve())])


if __name__ == "__main__":
    unittest.main()
