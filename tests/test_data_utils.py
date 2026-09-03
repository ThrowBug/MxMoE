import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed in this interpreter.") from error

from mxmoe.quant.data_utils import calibration_input_ids_sha256


class DataUtilsTests(unittest.TestCase):
    def test_calibration_hash_is_deterministic_and_dtype_independent(self):
        int32_samples = [torch.tensor([[1, 2, 3]], dtype=torch.int32)]
        int64_samples = [torch.tensor([[1, 2, 3]], dtype=torch.int64)]
        self.assertEqual(
            calibration_input_ids_sha256(int32_samples),
            calibration_input_ids_sha256(int64_samples),
        )


if __name__ == "__main__":
    unittest.main()
