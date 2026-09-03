"""Layer-wise mixed-bit GPTQ for standalone MxMoE fake-quant checkpoints."""

from __future__ import annotations

import gc
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from mxmoe.kernels.qconfig import QModelConfig
from mxmoe.quant.gemq_gptq import GPTQWeightQuantizer


EXPERT_LINEAR_RE = re.compile(
    r"(?:^|\.)mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
)
EXPERT_CONFIG_FIELD = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}


@dataclass(frozen=True)
class GPTQConfig:
    groupsize: int = 128
    blocksize: int = 128
    percdamp: float = 0.01
    mse: bool = True
    actorder: bool = False
    static_groups: bool = False
    attn_wbits: int = 4
    dense_wbits: int = 4

    def to_dict(self):
        return asdict(self)


def get_decoder_layers(model):
    return model.model.layers


def get_named_linears(module: nn.Module) -> dict[str, nn.Linear]:
    return {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    }


def get_linear_bit(
    name: str,
    layer_idx: int,
    qconfig: QModelConfig,
    config: GPTQConfig,
) -> int:
    """Classify a decoder Linear and return its assigned bit width."""
    if "self_attn" in name:
        return config.attn_wbits

    match = EXPERT_LINEAR_RE.search(name)
    if match:
        expert_idx, projection = match.groups()
        try:
            expert_cfg = qconfig.layers[str(layer_idx)].experts[expert_idx]
        except KeyError as error:
            raise KeyError(
                f"Missing allocation for layer {layer_idx}, expert {expert_idx}."
            ) from error
        return getattr(expert_cfg, EXPERT_CONFIG_FIELD[projection]).w_bits

    # Qwen3-MoE's router is mlp.gate and must stay in BF16.
    if name == "mlp.gate" or name.endswith(".mlp.gate"):
        return 16

    # A non-expert MLP is a dense FFN. Qwen3-30B-A3B currently has none,
    # but keeping this rule makes --dense-wbits explicit and testable.
    if "mlp" in name and ".experts." not in name:
        return config.dense_wbits

    return 16


def build_layer_bit_map(
    layer: nn.Module,
    layer_idx: int,
    qconfig: QModelConfig,
    config: GPTQConfig,
) -> dict[str, int]:
    return {
        name: get_linear_bit(name, layer_idx, qconfig, config)
        for name in get_named_linears(layer)
    }


def validate_qwen3_allocation(
    model,
    qconfig: QModelConfig,
    expected_groupsize: int = 128,
):
    layers = get_decoder_layers(model)
    if model.config.model_type != "qwen3_moe":
        raise ValueError(
            f"Expected model_type='qwen3_moe', got {model.config.model_type!r}."
        )
    if len(qconfig.layers) != len(layers):
        raise ValueError(
            f"Allocation has {len(qconfig.layers)} layers; model has {len(layers)}."
        )

    total_parameters = 0
    weighted_bits = 0
    bit_histogram: dict[int, dict[str, int]] = {}
    for layer_idx, layer in enumerate(layers):
        model_experts = len(layer.mlp.experts)
        allocated_experts = qconfig.layers[str(layer_idx)].experts
        if len(allocated_experts) != model_experts:
            raise ValueError(
                f"Layer {layer_idx} has {model_experts} experts but "
                f"{len(allocated_experts)} allocations."
            )
        for expert_idx, expert in enumerate(layer.mlp.experts):
            expert_cfg = allocated_experts[str(expert_idx)]
            for projection, field in EXPERT_CONFIG_FIELD.items():
                linear = getattr(expert, projection)
                linear_cfg = getattr(expert_cfg, field)
                bits = linear_cfg.w_bits
                if bits not in (1, 2, 3):
                    raise ValueError(
                        f"Unsupported expert bit width {bits} at layer {layer_idx}, "
                        f"expert {expert_idx}, {projection}."
                    )
                if (
                    linear_cfg.w_gsize != expected_groupsize
                    or linear_cfg.w_sym
                    or linear_cfg.a_bits != 16
                ):
                    raise ValueError(
                        "The comparison recipe requires asymmetric W1/W2/W3 "
                        f"G{expected_groupsize}-A16, but layer {layer_idx}, "
                        f"expert {expert_idx}, {projection} has {linear_cfg}."
                    )
                count = linear.weight.numel()
                total_parameters += count
                weighted_bits += bits * count
                bucket = bit_histogram.setdefault(bits, {"linears": 0, "parameters": 0})
                bucket["linears"] += 1
                bucket["parameters"] += count

    nominal_average = weighted_bits / total_parameters
    return {
        "layers": len(layers),
        "experts_per_layer": len(layers[0].mlp.experts),
        "expert_parameters": total_parameters,
        "nominal_average_bits": nominal_average,
        "effective_average_bits_g128": nominal_average + 0.25,
        "bit_histogram": bit_histogram,
    }


class _StopForward(RuntimeError):
    pass


@torch.no_grad()
def capture_first_layer_inputs(model, dataloader, device: str = "cuda"):
    """Capture X_0 and decoder kwargs without executing the full model."""
    layers = get_decoder_layers(model)
    inputs = []
    layer_kwargs: dict[str, Any] = {}

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, hidden_states, **kwargs):
            inputs.append(hidden_states.detach())
            layer_kwargs.update(kwargs)
            raise _StopForward

    layers[0] = Catcher(layers[0])
    try:
        for sample in tqdm(dataloader, desc="Capturing decoder inputs"):
            try:
                model(sample.to(device))
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].module

    layer_inputs = torch.cat(inputs, dim=0)
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    layers[0] = layers[0].cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return layer_inputs, layer_kwargs


def _make_hessian_hook(quantizer: GPTQWeightQuantizer):
    def hook(_module, inputs, _output):
        quantizer.add_batch(inputs[0].detach())

    return hook


@torch.no_grad()
def quantize_model_mixed_gptq(
    model,
    dataloader,
    qconfig: QModelConfig,
    config: GPTQConfig,
    device: str = "cuda",
):
    """Quantize each decoder layer and advance calibration state once per layer."""
    if config.actorder or config.static_groups:
        raise ValueError(
            "The GEMQ-aligned recipe requires actorder=False and static_groups=False."
        )
    allocation_summary = validate_qwen3_allocation(
        model,
        qconfig,
        expected_groupsize=config.groupsize,
    )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)
    layer_inputs, layer_kwargs = capture_first_layer_inputs(model, dataloader, device)
    layer_outputs = torch.zeros_like(layer_inputs)
    quantized_linears = []

    try:
        for layer_idx in tqdm(range(len(layers)), desc="Mixed-bit GPTQ"):
            layer = layers[layer_idx].to(device)
            named_linears = get_named_linears(layer)
            bit_map = build_layer_bit_map(layer, layer_idx, qconfig, config)
            quantizers: dict[str, GPTQWeightQuantizer] = {}
            handles = []

            for name, linear in named_linears.items():
                bits = bit_map[name]
                if bits >= 16:
                    continue
                columns = linear.weight.shape[1]
                if columns % config.groupsize == 0:
                    groupsize = config.groupsize
                elif columns % 64 == 0:
                    groupsize = 64
                else:
                    raise ValueError(
                        f"Neither G{config.groupsize} nor G64 divides {name}'s "
                        f"input dimension {columns}."
                    )
                quantizer = GPTQWeightQuantizer(
                    linear.weight.data,
                    name=f"model.layers.{layer_idx}.{name}",
                    nbits=bits,
                    blocksize=config.blocksize,
                    percdamp=config.percdamp,
                    groupsize=groupsize,
                    actorder=config.actorder,
                    static_groups=config.static_groups,
                    mse=config.mse,
                )
                quantizers[name] = quantizer
                handles.append(linear.register_forward_hook(_make_hessian_hook(quantizer)))

            # All hooks observe the same immutable X_l. Outputs from this pass are
            # deliberately not assigned back to layer_inputs.
            try:
                for sample_idx in range(layer_inputs.shape[0]):
                    layer_outputs[sample_idx] = layer(
                        layer_inputs[sample_idx : sample_idx + 1], **layer_kwargs
                    )[0]
            finally:
                for handle in handles:
                    handle.remove()

            for name, linear in named_linears.items():
                quantizer = quantizers.get(name)
                if quantizer is None:
                    continue
                qweight, scales, zeros = quantizer.quantize()
                restored = quantizer.dequantize(qweight, scales, zeros)
                linear.weight.data = restored.reshape_as(linear.weight.data)
                quantized_linears.append(
                    {
                        "name": f"model.layers.{layer_idx}.{name}",
                        "bits": quantizer.nbits,
                        "groupsize": quantizer.groupsize,
                        "parameters": linear.weight.numel(),
                    }
                )

            # The only state transition for this decoder layer: X_l -> X_(l+1).
            for sample_idx in range(layer_inputs.shape[0]):
                layer_outputs[sample_idx] = layer(
                    layer_inputs[sample_idx : sample_idx + 1], **layer_kwargs
                )[0]
            layer_inputs, layer_outputs = layer_outputs, layer_inputs
            layers[layer_idx] = layer.cpu()
            del layer, quantizers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model.config.use_cache = use_cache

    return {
        "allocation": allocation_summary,
        "gptq": config.to_dict(),
        "quantized_linears": quantized_linears,
    }
