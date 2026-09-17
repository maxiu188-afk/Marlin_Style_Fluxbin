import json
import unittest
from pathlib import Path

import torch

from fluxbin_style.hierarchical_w2 import (
    choose_linear_steps,
    hierarchical_gptq_quantize,
    make_payload,
    materialize_payload,
    nominal_bpw,
    pack_unsigned,
    project_hierarchical_group,
    unpack_unsigned,
)


ROOT = Path(__file__).resolve().parents[1]


class HierarchicalW2Test(unittest.TestCase):
    def test_unsigned_pack_round_trip(self) -> None:
        for bits in (2, 4, 8):
            count = 37
            values = (torch.arange(count, dtype=torch.int64) % (1 << bits)).to(torch.uint8)
            packed = pack_unsigned(values, bits)
            decoded = unpack_unsigned(packed, bits, count=count, shape=(count,))
            self.assertTrue(torch.equal(values, decoded))

    def test_requested_nominal_rates(self) -> None:
        self.assertEqual(nominal_bpw(4, 4), 2.5)
        self.assertEqual(nominal_bpw(8, 4), 2.625)
        self.assertEqual(nominal_bpw(4, 8), 2.75)
        self.assertEqual(nominal_bpw(8, 8), 2.875)

    def test_projection_uses_only_w2_levels_and_fp16_parent(self) -> None:
        generator = torch.Generator().manual_seed(12)
        weight = torch.randn((7, 128), generator=generator)
        r32_step, r16_step = choose_linear_steps(weight, r32_bits=4, r16_bits=8)
        result = project_hierarchical_group(
            weight,
            r32_bits=4,
            r16_bits=8,
            r32_step=r32_step,
            r16_step=r16_step,
        )
        self.assertEqual(set(result.q.unique().tolist()), {-3.0, -1.0, 1.0, 3.0})
        self.assertTrue(torch.equal(result.parent, result.parent.to(torch.float16).to(torch.float32)))
        self.assertTrue(torch.isfinite(result.reconstructed).all())
        self.assertLess(torch.mean((weight - result.reconstructed).square()).item(), torch.mean(weight.square()).item())

    def test_payload_materialization_and_storage_accounting(self) -> None:
        rows, columns = 2, 256
        groups = columns // 128
        q = torch.tensor([-3, -1, 1, 3], dtype=torch.int8).repeat(rows, columns // 4)
        parent = torch.ones((rows, groups), dtype=torch.float32)
        r32 = torch.full((rows, groups, 4), 8, dtype=torch.uint8)
        r16 = torch.full((rows, groups, 8), 8, dtype=torch.uint8)
        permutation = torch.arange(columns - 1, -1, -1)
        payload = make_payload(
            q,
            parent,
            r32,
            r16,
            r32_bits=4,
            r16_bits=4,
            r32_step=torch.tensor(0.125),
            r16_step=torch.tensor(0.125),
            permutation=permutation,
        )
        decoded = materialize_payload(payload)
        self.assertEqual(tuple(decoded.shape), (rows, columns))
        expected_bytes = (
            rows * columns * 2 // 8
            + rows * groups * 2
            + rows * groups * 4 * 4 // 8
            + rows * groups * 8 * 4 // 8
            + 2
            + 2
            + columns * 4
        )
        self.assertEqual(payload.tensor_bytes, expected_bytes)
        self.assertEqual(payload.actual_bpw, expected_bytes * 8 / (rows * columns))

    def test_identity_hessian_gptq_round_trip_matches_payload(self) -> None:
        generator = torch.Generator().manual_seed(27)
        weight = torch.randn((5, 128), generator=generator)
        diagonal = torch.linspace(0.5, 3.0, 128)
        result = hierarchical_gptq_quantize(
            weight,
            torch.diag(diagonal),
            r32_bits=4,
            r16_bits=4,
        )
        decoded = materialize_payload(result.payload)
        self.assertTrue(torch.equal(decoded, result.reconstructed))
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertEqual(sorted(result.payload.permutation.tolist()), list(range(128)))
        self.assertGreater(result.squared_error, 0.0)

    def test_frozen_config_has_only_four_arms_and_exclusions(self) -> None:
        path = ROOT / "configs/evaluation/qwen3_8b_hierarchical_w2_v1.json"
        config = json.loads(path.read_text())
        self.assertEqual(list(config["variants"]), ["H2.50", "H2.625", "H2.75", "H2.875"])
        self.assertEqual(
            config["evaluation"]["arms"],
            ["gptq_w3_g128_sym", "H2.50", "H2.625", "H2.75", "H2.875"],
        )
        self.assertTrue(all(config["policy"].values()))
        self.assertTrue(config["hierarchical_projection"]["direct_gptq_projection"])
        self.assertEqual(config["evaluation"]["other_accuracy_metrics"], [])


if __name__ == "__main__":
    unittest.main()
