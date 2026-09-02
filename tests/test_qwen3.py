import unittest

from fluxbin_style import (
    build_qwen3_linear_inventory,
    expected_qwen3_linear_shape,
    qwen3_linear_weight_names,
)


class Qwen3InventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }

    def test_inventory_covers_seven_linears_per_layer(self) -> None:
        records = []
        for name in qwen3_linear_weight_names(2):
            module = name.split(".", 3)[3].removesuffix(".weight")
            records.append(
                {"name": name, "shape": expected_qwen3_linear_shape(self.config, module)}
            )
        records.append({"name": "model.layers.0.input_layernorm.weight", "shape": [16]})
        result = build_qwen3_linear_inventory(self.config, records)
        self.assertEqual(result["included_tensor_count"], 14)
        self.assertEqual(result["excluded_tensor_count"], 1)

    def test_missing_linear_is_rejected(self) -> None:
        records = []
        for name in qwen3_linear_weight_names(2)[:-1]:
            module = name.split(".", 3)[3].removesuffix(".weight")
            records.append(
                {"name": name, "shape": expected_qwen3_linear_shape(self.config, module)}
            )
        with self.assertRaises(ValueError):
            build_qwen3_linear_inventory(self.config, records)


if __name__ == "__main__":
    unittest.main()
