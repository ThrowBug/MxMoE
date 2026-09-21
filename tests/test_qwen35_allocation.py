import unittest

from mxmoe.quant.qwen35_allocate import allocate_layer, validate_run_metadata


class Qwen35AllocationTests(unittest.TestCase):
    def test_exact_budget_and_gate_up_pairing(self):
        losses = {
            1: [[9, 9, 9], [1, 1, 1]],
            2: [[3, 3, 3], [0.5, 0.5, 0.5]],
            3: [[0, 0, 0], [0, 0, 0]],
        }
        choices, objective = allocate_layer(losses, nominal_bits=2.0)
        self.assertEqual(len(choices), 2)
        self.assertEqual(sum(2 * gate + down for gate, down in choices), 12)
        self.assertGreaterEqual(objective, 0)

    def test_impossible_fractional_budget_rejected(self):
        losses = {bit: [[0.0, 0.0, 0.0]] for bit in (1, 2, 3)}
        with self.assertRaises(ValueError):
            allocate_layer(losses, nominal_bits=1.1)

    def test_run_metadata_rejects_mixed_models_and_settings(self):
        common = {"model": "example", "fixed_config": {"attn_wbits": 4}}
        self.assertEqual(validate_run_metadata(
            [("trace", common), ("loss", dict(common))], "example"
        ), common["fixed_config"])
        with self.assertRaises(ValueError):
            validate_run_metadata(
                [("trace", common), ("loss", {**common, "model": "other"})],
                "example",
            )
        with self.assertRaises(ValueError):
            validate_run_metadata(
                [("trace", common), ("loss", {**common, "fixed_config": {"attn_wbits": 16}})],
                "example",
            )


if __name__ == "__main__":
    unittest.main()
