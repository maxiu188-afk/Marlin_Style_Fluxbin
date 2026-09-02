import unittest

import torch

from fluxbin_style import pack_two_bases, unpack_two_bases


class TwoBasePackingTests(unittest.TestCase):
    def test_round_trip_random_bases(self) -> None:
        torch.manual_seed(31)
        bases = torch.where(
            torch.rand(2, 7, 20) < 0.5,
            torch.tensor(-1, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        )
        codes = pack_two_bases(bases)
        self.assertEqual(codes.dtype, torch.uint8)
        self.assertEqual(tuple(codes.shape), (7, 5))
        torch.testing.assert_close(unpack_two_bases(codes), bases)

    def test_known_interleaved_bit_order(self) -> None:
        bases = torch.tensor(
            [
                [[-1, 1, -1, 1]],
                [[-1, -1, 1, 1]],
            ],
            dtype=torch.int8,
        )
        # K0..K3 produce bit pairs 00, 01, 10, 11 from LSB to MSB.
        codes = pack_two_bases(bases)
        self.assertEqual(int(codes.item()), 0b11100100)
        torch.testing.assert_close(unpack_two_bases(codes), bases)

    def test_rejects_non_binary_input(self) -> None:
        with self.assertRaises(ValueError):
            pack_two_bases(torch.zeros(2, 3, 8, dtype=torch.int8))


if __name__ == "__main__":
    unittest.main()
