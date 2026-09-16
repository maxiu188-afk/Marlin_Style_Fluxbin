import unittest
from unittest.mock import patch

import torch

from fluxbin_style.gptq_deployment import (
    bf16_weight_semantics_w3_matvec,
    convert_gptq_w3_to_planar,
    structural_w3_matvec,
)
from fluxbin_style.w3_lut_deployment import (
    EXPERIMENTAL_ROW_TILES,
    ROW_TILES,
    load_w3_lut_extension,
    w3_lut_m1_out,
    workspace_shape,
)
from test_gptq_w3_planar import synthetic_raw


class W3LUTInterfaceTest(unittest.TestCase):
    def test_workspace_and_extension_variants(self):
        self.assertEqual(workspace_shape(1024, 4096), (32, 1024))
        self.assertEqual(workspace_shape(1024, 4096, 4), (8, 1024))
        self.assertEqual(workspace_shape(1024, 4096, 32), (1, 1024))
        with self.assertRaises(ValueError):
            workspace_shape(1024, 4097)
        with self.assertRaises(ValueError):
            workspace_shape(1024, 4096, 0)
        load_w3_lut_extension.cache_clear()
        try:
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.utils.cpp_extension.load"
            ) as build:
                for row_tile in EXPERIMENTAL_ROW_TILES:
                    load_w3_lut_extension(row_tile)
                    flags = build.call_args.kwargs["extra_cuda_cflags"]
                    self.assertIn(f"-DW3_LUT_ROWS={row_tile}", flags)
                    self.assertTrue(build.call_args.kwargs["sources"][0].endswith("w3_lut.cu"))
                for row_tile in EXPERIMENTAL_ROW_TILES:
                    load_w3_lut_extension(row_tile)
                self.assertEqual(build.call_count, 4)
                with self.assertRaises(ValueError):
                    load_w3_lut_extension(128)
        finally:
            load_w3_lut_extension.cache_clear()


@unittest.skipUnless(torch.cuda.is_available(), "requires NVIDIA CUDA compiler/device")
class CUDAW3LUTCorrectnessTest(unittest.TestCase):
    def test_inline_all_row_tiles_fp16_bf16(self):
        for out_features, in_features in ((32, 128), (64, 256), (288, 384)):
            raw, _, _ = synthetic_raw(k=in_features, o=out_features)
            cpu_layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
            layout = {name: value.cuda() for name, value in cpu_layout.items()}
            for dtype in (torch.float16, torch.bfloat16):
                torch.cuda.manual_seed_all(20260916)
                x = torch.randn(1, in_features, device="cuda", dtype=dtype)
                reference = structural_w3_matvec(x, layout)
                for row_tile in ROW_TILES:
                    out = torch.empty(1, out_features, device="cuda", dtype=dtype)
                    workspace = torch.full(
                        workspace_shape(out_features, in_features),
                        float("nan"),
                        device="cuda",
                    )

                    def run():
                        return w3_lut_m1_out(x, layout, out, workspace, row_tile=row_tile)

                    run()
                    self.assertTrue(torch.isfinite(workspace).all())
                    torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.02)
                    saved = out.clone()
                    workspace.fill_(float("nan"))
                    run()
                    self.assertTrue(torch.equal(out, saved))

                    graph = torch.cuda.CUDAGraph()
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        run()
                    graph.replay()
                    self.assertTrue(torch.equal(out, saved))

                if dtype == torch.bfloat16:
                    corrected_reference = bf16_weight_semantics_w3_matvec(
                        x, layout
                    ).bfloat16()
                    for groups_per_split in (1, in_features // 128):
                        corrected = torch.empty_like(corrected_reference)
                        corrected_workspace = torch.full(
                            workspace_shape(
                                out_features,
                                in_features,
                                groups_per_split,
                            ),
                            float("nan"),
                            device="cuda",
                        )
                        w3_lut_m1_out(
                            x,
                            layout,
                            corrected,
                            corrected_workspace,
                            row_tile=256,
                            groups_per_split=groups_per_split,
                            arithmetic="decoded_bf16",
                        )
                        torch.testing.assert_close(
                            corrected.float(),
                            corrected_reference.float(),
                            rtol=0.005,
                            atol=0.02,
                        )
                        if groups_per_split == 1:
                            self.assertTrue(torch.isfinite(corrected_workspace).all())
                        saved = corrected.clone()
                        w3_lut_m1_out(
                            x,
                            layout,
                            corrected,
                            corrected_workspace,
                            row_tile=256,
                            groups_per_split=groups_per_split,
                            arithmetic="decoded_bf16",
                        )
                        self.assertTrue(torch.equal(corrected, saved))


if __name__ == "__main__":
    unittest.main()
