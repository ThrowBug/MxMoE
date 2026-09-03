import unittest

from project_config import ID2NAME


class ProjectConfigTests(unittest.TestCase):
    def test_qwen3_instruct_2507_alias_uses_full_model_name(self):
        self.assertEqual(
            ID2NAME["qwen3_moe_30b_a3b_instruct_2507"],
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        )


if __name__ == "__main__":
    unittest.main()
