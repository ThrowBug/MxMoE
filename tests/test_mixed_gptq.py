from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed in this interpreter.") from error

from mxmoe.kernels.qconfig import (
    QExpertConfig,
    QLayerConfig,
    QLinearConfig,
    QModelConfig,
)
from mxmoe.quant import mixed_gptq


class ToyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class ToyExpert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class ToyMlp(nn.Module):
    def __init__(self, width, num_experts):
        super().__init__()
        self.gate = nn.Linear(width, num_experts, bias=False)
        self.experts = nn.ModuleList([ToyExpert(width) for _ in range(num_experts)])

    def forward(self, x):
        # Exercise every expert so every hook receives calibration samples.
        return sum(expert(x) for expert in self.experts) / len(self.experts)


class ToyLayer(nn.Module):
    def __init__(self, width, num_experts):
        super().__init__()
        self.self_attn = ToyAttention(width)
        self.mlp = ToyMlp(width, num_experts)
        self.forward_calls = 0

    def forward(self, hidden_states, **_kwargs):
        self.forward_calls += 1
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return (hidden_states + self.mlp(hidden_states),)


class ToyBackbone(nn.Module):
    def __init__(self, width, layers, experts):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, width)
        self.layers = nn.ModuleList(
            [ToyLayer(width, experts) for _ in range(layers)]
        )


class ToyModel(nn.Module):
    def __init__(self, width=4, layers=2, experts=2):
        super().__init__()
        self.model = ToyBackbone(width, layers, experts)
        self.config = SimpleNamespace(model_type="qwen3_moe", use_cache=True)

    def forward(self, input_ids):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return (hidden_states,)


class IdentityGPTQ:
    instances = []

    def __init__(self, weight, name, nbits, groupsize, **_kwargs):
        self.W = weight.clone()
        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize
        self.inputs = []
        self.__class__.instances.append(self)

    def add_batch(self, inp):
        self.inputs.append(inp.clone())

    def quantize(self):
        qweight = self.W.reshape(-1, self.groupsize)
        groups = qweight.shape[0]
        scale = torch.ones(groups, 1, dtype=qweight.dtype)
        zero = torch.zeros_like(scale)
        return qweight, scale, zero

    @staticmethod
    def dequantize(qweight, scales, zeros):
        return (qweight - zeros) * scales


def build_qconfig(num_layers=2, num_experts=2):
    qlinear = QLinearConfig(w_bits=2, w_gsize=2, w_sym=False)
    return QModelConfig(
        layers={
            str(layer): QLayerConfig(
                experts={
                    str(expert): QExpertConfig(qlinear, qlinear, qlinear)
                    for expert in range(num_experts)
                }
            )
            for layer in range(num_layers)
        }
    )


class MixedGPTQTests(unittest.TestCase):
    def test_layer_state_advances_once_after_all_hooks(self):
        IdentityGPTQ.instances = []
        with patch.object(mixed_gptq, "GPTQWeightQuantizer", IdentityGPTQ):
            model = ToyModel()
            samples = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
            report = mixed_gptq.quantize_model_mixed_gptq(
                model,
                samples,
                build_qconfig(),
                mixed_gptq.GPTQConfig(
                    groupsize=2,
                    blocksize=2,
                    attn_wbits=4,
                    dense_wbits=4,
                ),
                device="cpu",
            )

        # Per layer: one pass to collect all Hessians and one post-quant pass.
        # The old bug instead performed one full layer pass per Linear.
        self.assertEqual(
            [layer.forward_calls for layer in model.model.layers], [4, 4]
        )
        self.assertTrue(
            all(len(instance.inputs) == len(samples) for instance in IdentityGPTQ.instances)
        )
        self.assertEqual(report["allocation"]["nominal_average_bits"], 2.0)
        self.assertEqual(report["allocation"]["effective_average_bits_g128"], 2.25)
        self.assertTrue(model.config.use_cache)

    def test_qwen3_linear_classification_keeps_router_in_bfloat16(self):
        model = ToyModel(layers=1)
        bit_map = mixed_gptq.build_layer_bit_map(
            model.model.layers[0],
            0,
            build_qconfig(num_layers=1),
            mixed_gptq.GPTQConfig(groupsize=2),
        )
        self.assertEqual(bit_map["mlp.gate"], 16)
        self.assertEqual(bit_map["self_attn.q_proj"], 4)
        self.assertEqual(bit_map["mlp.experts.0.gate_proj"], 2)


if __name__ == "__main__":
    unittest.main()
