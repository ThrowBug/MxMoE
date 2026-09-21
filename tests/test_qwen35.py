from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed in this interpreter.") from error

from mxmoe.kernels.qconfig import QExpertConfig, QLayerConfig, QLinearConfig, QModelConfig
from mxmoe.quant.qwen35_allocate import allocate_layer
from mxmoe.quant.qwen35_core import (
    expert_weight, fixed_linears, kwargs_for_layer, selected_tokens, validate_allocation,
)


class PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.arange(2 * 8 * 4).float().reshape(2, 8, 4))
        self.down_proj = nn.Parameter(torch.ones(2, 4, 4))


class ToyLayer(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(4, 4, bias=False))
        self.linear_attn = nn.Module()
        for name in ("in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b"):
            setattr(self.linear_attn, name, nn.Linear(4, 4, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(4, 2, bias=False)
        self.mlp.shared_expert_gate = nn.Linear(4, 1, bias=False)
        self.mlp.shared_expert = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp.shared_expert, name, nn.Linear(4, 4, bias=False))
        self.mlp.experts = PackedExperts()


class CoreTests(unittest.TestCase):
    def test_packed_gate_up_down_slices(self):
        experts = PackedExperts()
        self.assertEqual(tuple(expert_weight(experts, 1, "gate").shape), (4, 4))
        self.assertEqual(tuple(expert_weight(experts, 1, "up").shape), (4, 4))
        self.assertEqual(tuple(expert_weight(experts, 1, "down").shape), (4, 4))
        with torch.no_grad():
            expert_weight(experts, 1, "up").fill_(123)
        self.assertTrue(torch.all(experts.gate_up_proj[1, 4:] == 123))
        self.assertFalse(torch.any(experts.gate_up_proj[0] == 123))

    def test_fixed_selection_excludes_router_and_other_linear_attention(self):
        settings = {"attn_wbits": 4, "linear_attn_wbits": 16, "shared_expert_wbits": 4}
        linear_names = fixed_linears(ToyLayer("linear_attention"), settings)
        self.assertEqual(set(linear_names), {
            "linear_attn.in_proj_qkv", "linear_attn.out_proj",
            "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
        })
        self.assertEqual(linear_names["linear_attn.out_proj"][1], 16)
        full_names = fixed_linears(ToyLayer("full_attention"), settings)
        self.assertIn("self_attn.q_proj", full_names)
        self.assertNotIn("mlp.gate", full_names)
        self.assertNotIn("mlp.shared_expert_gate", full_names)

    def test_layer_mask_dispatch(self):
        common = {"position_embeddings": "rope", "position_ids": "ids",
                  "cache_position": "cache", "causal_mask": "causal"}
        self.assertIsNone(kwargs_for_layer(ToyLayer("linear_attention"), common)["attention_mask"])
        self.assertEqual(kwargs_for_layer(ToyLayer("full_attention"), common)["attention_mask"], "causal")

    def test_selected_tokens_preserve_router_weights(self):
        hidden = [torch.arange(12).reshape(3, 4).float()]
        ids = [torch.tensor([[0, 1], [1, 0], [0, 1]])]
        scores = [torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])]
        x, w = selected_tokens(hidden, ids, scores, 1, "cpu")
        self.assertTrue(torch.equal(x, hidden[0]))
        self.assertTrue(torch.allclose(w, torch.tensor([0.8, 0.4, 0.3])))

    def test_allocation_counts_only_routed_experts(self):
        layer = ToyLayer("linear_attention")
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            model=SimpleNamespace(layers=[layer]),
        )
        q = QLinearConfig(w_bits=2, w_gsize=2)
        cfg = QModelConfig(layers={"0": QLayerConfig(experts={
            "0": QExpertConfig(q, q, q), "1": QExpertConfig(q, q, q)
        })})
        summary = validate_allocation(model, cfg, groupsize=2)
        self.assertEqual(summary["experts_per_layer"], 2)
        self.assertEqual(summary["expert_parameters"], 2 * 3 * 16)
        self.assertEqual(summary["nominal_average_bits"], 2.0)
        cfg.layers["0"].experts["0"].up = QLinearConfig(w_bits=3, w_gsize=2)
        with self.assertRaises(ValueError):
            validate_allocation(model, cfg, groupsize=2)

    def test_exact_budget_dp_keeps_gate_and_up_together(self):
        losses = {
            1: [[9, 9, 9], [1, 1, 1]],
            2: [[3, 3, 3], [0.5, 0.5, 0.5]],
            3: [[0, 0, 0], [0, 0, 0]],
        }
        choices, _ = allocate_layer(losses, nominal_bits=2.0)
        self.assertEqual(sum(2 * gate + down for gate, down in choices), 12)
        self.assertEqual(len(choices), 2)


if __name__ == "__main__":
    unittest.main()
