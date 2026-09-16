import unittest

import torch

from fluxbin_style.gptq_deployment import (
    FORMAT,
    convert_gptq_w3_to_planar,
    decode_planar_w3_codes,
    inspect_gptq_checkpoint,
    layout_storage,
    pack_gptq_w3_codes,
    restore_planar_w3,
    structural_w3_matvec,
    unpack_gptq_w3_qweight,
    unpack_gptq_w3_qzeros,
    validate_planar_w3,
)


def synthetic_raw(*, k=256, o=64, seed=20260916, zero=4):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 8, (k, o), generator=generator, dtype=torch.uint8)
    groups = k // 128
    zeros = torch.full((groups, o), zero, dtype=torch.uint8)
    scales = (torch.rand((groups, o), generator=generator) * 0.125 + 0.001).half()
    # Mimic desc_act=True: every original K position retains the quantization
    # group assigned in activation-order space.
    activation_order = torch.randperm(k, generator=generator)
    g_idx = torch.empty(k, dtype=torch.int32)
    g_idx[activation_order] = torch.arange(groups, dtype=torch.int32).repeat_interleave(128)
    raw = {
        "qweight": pack_gptq_w3_codes(codes),
        # Raw FORMAT.GPTQ stores logical zeros minus one.
        "qzeros": pack_gptq_w3_codes(((zeros.to(torch.int16) - 1) & 7).to(torch.uint8), packed_axis=1),
        "scales": scales,
        "g_idx": g_idx,
    }
    return raw, codes, zeros


class GPTQW3PlanarTest(unittest.TestCase):
    def test_unpack_matches_independent_canonical_boundary_layout(self):
        codes = torch.arange(32, dtype=torch.uint8).remainder_(8).reshape(32, 1)
        words = [0, 0, 0]
        for index in range(10):
            words[0] |= int(codes[index]) << (3 * index)
        words[0] |= (int(codes[10]) & 3) << 30
        words[1] |= int(codes[10]) >> 2
        for index in range(11, 21):
            words[1] |= int(codes[index]) << (1 + 3 * (index - 11))
        words[1] |= (int(codes[21]) & 1) << 31
        words[2] |= int(codes[21]) >> 1
        for index in range(22, 32):
            words[2] |= int(codes[index]) << (2 + 3 * (index - 22))
        signed = torch.tensor(
            [value if value < 2**31 else value - 2**32 for value in words],
            dtype=torch.int32,
        ).reshape(3, 1)
        self.assertTrue(torch.equal(unpack_gptq_w3_qweight(signed), codes))

    def test_canonical_pack_unpack(self):
        raw, codes, zeros = synthetic_raw()
        self.assertTrue(torch.equal(unpack_gptq_w3_qweight(raw["qweight"]), codes))
        self.assertTrue(torch.equal(unpack_gptq_w3_qzeros(raw["qzeros"], qzero_format=1), zeros))

        # A loaded GPTQModel TorchLinear has already converted qzero v1 to v2.
        v2 = pack_gptq_w3_codes(zeros, packed_axis=1)
        self.assertTrue(torch.equal(unpack_gptq_w3_qzeros(v2, qzero_format=2), zeros))

    def test_desc_act_canonicalization_is_exact(self):
        raw, codes, _ = synthetic_raw()
        facts = inspect_gptq_checkpoint(raw, qzero_format=1)
        self.assertEqual(facts["decoded_zero_unique"], [4])
        self.assertFalse(facts["permutation_identity"])
        layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
        self.assertEqual(set(layout), {"planes", "scales", "perm"})
        self.assertEqual(layout["perm"].dtype, torch.int16)
        self.assertTrue(torch.equal(decode_planar_w3_codes(layout), codes))

    def test_bf16_restore_matches_gptq_semantics_bitwise(self):
        raw, codes, zeros = synthetic_raw()
        layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
        expected = (
            raw["scales"][raw["g_idx"].long()]
            * (codes.to(torch.int16) - zeros[raw["g_idx"].long()])
        ).T.to(torch.bfloat16).contiguous()
        self.assertTrue(torch.equal(restore_planar_w3(layout), expected))

    def test_structural_matvec_matches_direct_formula(self):
        raw, codes, zeros = synthetic_raw()
        layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
        x = torch.randn(1, codes.shape[0], generator=torch.Generator().manual_seed(9), dtype=torch.bfloat16)
        direct = torch.sum(
            x.float().T
            * raw["scales"][raw["g_idx"].long()].float()
            * (codes.to(torch.int16) - zeros[raw["g_idx"].long()]).float(),
            dim=0,
        ).unsqueeze(0)
        actual = structural_w3_matvec(x, layout)
        torch.testing.assert_close(actual, direct, rtol=2e-6, atol=2e-5)

    def test_layout_storage_includes_permutation(self):
        raw, _, _ = synthetic_raw()
        record = layout_storage(convert_gptq_w3_to_planar(raw, qzero_format=1))
        self.assertEqual(record["format"], FORMAT)
        expected = 3 * 256 * 64 // 8 + 2 * 64 * 2 + 2 * 256
        self.assertEqual(record["total_bytes"], expected)
        self.assertAlmostEqual(record["bits_per_weight"], expected * 8 / (256 * 64))

    def test_non_constant_zero_rejected_by_v1(self):
        raw, _, _ = synthetic_raw(zero=3)
        with self.assertRaisesRegex(ValueError, "requires decoded zero=4"):
            convert_gptq_w3_to_planar(raw, qzero_format=1)

    def test_runtime_validation_skips_only_value_checks(self):
        raw, _, _ = synthetic_raw()
        layout = convert_gptq_w3_to_planar(raw, qzero_format=1)
        layout["perm"] = torch.zeros_like(layout["perm"])
        with self.assertRaisesRegex(ValueError, "bijection"):
            validate_planar_w3(layout)
        self.assertEqual(validate_planar_w3(layout, check_values=False), (64, 256))


if __name__ == "__main__":
    unittest.main()
