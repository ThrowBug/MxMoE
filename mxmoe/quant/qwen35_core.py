"""Text-only Qwen3.5 MoE layer replay and packed-expert GPTQ helpers.

This module deliberately does not alter the existing Qwen3 quantizer.  The
Qwen3.5 text model has per-expert slices in two 3-D parameters, not Linear
submodules, and its two attention types use different masks.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask

from mxmoe.quant.gemq_gptq import GPTQWeightQuantizer


FIXED_KEYS = ("attn_wbits", "linear_attn_wbits", "shared_expert_wbits")


def check_fixed_bits(settings: dict) -> None:
    for key in FIXED_KEYS:
        if settings[key] not in (4, 16):
            raise ValueError(f"{key} must be 4 or 16, got {settings[key]}.")


def check_text_model(model) -> None:
    if model.config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expected the text-only Qwen3.5 MoE CausalLM, not the multimodal model.")
    if not hasattr(model.model, "layers"):
        raise ValueError("Qwen3.5 text decoder layers are unavailable.")
    for layer in model.model.layers:
        experts = layer.mlp.experts
        if experts.gate_up_proj.ndim != 3 or experts.down_proj.ndim != 3:
            raise ValueError("Expected packed three-dimensional routed-expert weights.")
        if experts.gate_up_proj.shape[0] != experts.down_proj.shape[0]:
            raise ValueError("Packed expert counts disagree.")
        if experts.gate_up_proj.shape[1] != 2 * experts.down_proj.shape[2]:
            raise ValueError("Packed gate/up split does not match down projection.")
        if experts.gate_up_proj.shape[2] != experts.down_proj.shape[1]:
            raise ValueError("Packed expert hidden dimensions disagree.")


def fixed_linears(layer, settings: dict) -> dict[str, tuple[nn.Linear, int]]:
    selected = {}
    if layer.layer_type == "full_attention":
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            selected[f"self_attn.{name}"] = (getattr(layer.self_attn, name), settings["attn_wbits"])
    elif layer.layer_type == "linear_attention":
        for name in ("in_proj_qkv", "out_proj"):
            selected[f"linear_attn.{name}"] = (getattr(layer.linear_attn, name), settings["linear_attn_wbits"])
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer type {layer.layer_type!r}.")
    for name in ("gate_proj", "up_proj", "down_proj"):
        selected[f"mlp.shared_expert.{name}"] = (
            getattr(layer.mlp.shared_expert, name), settings["shared_expert_wbits"]
        )
    return selected


def expert_weight(experts, expert_idx: int, projection: str) -> torch.Tensor:
    width = experts.down_proj.shape[2]
    if projection == "gate":
        return experts.gate_up_proj[expert_idx, :width]
    if projection == "up":
        return experts.gate_up_proj[expert_idx, width:]
    if projection == "down":
        return experts.down_proj[expert_idx]
    raise ValueError(f"Unknown expert projection {projection!r}.")


def validate_allocation(model, qconfig, groupsize: int = 128) -> dict:
    check_text_model(model)
    layers = model.model.layers
    if set(qconfig.layers) != {str(i) for i in range(len(layers))}:
        raise ValueError("Allocation layer IDs do not match the text model.")
    hist = {}
    weighted = total = 0
    for layer_idx, layer in enumerate(layers):
        experts = layer.mlp.experts
        count = experts.gate_up_proj.shape[0]
        allocations = qconfig.layers[str(layer_idx)].experts
        if set(allocations) != {str(e) for e in range(count)}:
            raise ValueError(f"Layer {layer_idx}: expected exactly {count} routed experts.")
        for expert_idx in range(count):
            expert_cfg = allocations[str(expert_idx)]
            if expert_cfg.gate.w_bits != expert_cfg.up.w_bits:
                raise ValueError(f"Layer {layer_idx}, expert {expert_idx}: gate/up must share one bit width.")
            for name, cfg in expert_cfg.qmap().items():
                if cfg.w_bits not in (1, 2, 3) or cfg.w_gsize != groupsize or cfg.w_sym or cfg.a_bits != 16:
                    raise ValueError(f"Invalid routed-expert recipe at layer {layer_idx}, expert {expert_idx}, {name}.")
                n = expert_weight(experts, expert_idx, name).numel()
                hist[cfg.w_bits] = hist.get(cfg.w_bits, 0) + n
                weighted += n * cfg.w_bits
                total += n
    nominal = weighted / total
    return {
        "layers": len(layers), "experts_per_layer": layers[0].mlp.experts.gate_up_proj.shape[0],
        "expert_parameters": total, "nominal_average_bits": nominal,
        "effective_average_bits_g128": nominal + 0.25, "bit_histogram_parameters": hist,
    }


@dataclass(frozen=True)
class ReplaySettings:
    device: str = "cuda:0"
    groupsize: int = 128
    blocksize: int = 128
    percdamp: float = 0.01


def replay_kwargs(model, sequence_length: int, device: str) -> dict:
    """Replicate Qwen3.5 text model's no-padding/no-cache prefill arguments."""
    config = model.config
    cache_position = torch.arange(sequence_length, device=device)
    position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
    hidden = torch.empty((1, sequence_length, config.hidden_size), device=device, dtype=next(model.parameters()).dtype)
    model.model.rotary_emb = model.model.rotary_emb.to(device)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    causal_mask = create_causal_mask(
        config=config, inputs_embeds=hidden, attention_mask=None,
        cache_position=cache_position, past_key_values=None,
        position_ids=position_ids[0],
    )
    return {
        "position_embeddings": position_embeddings, "position_ids": position_ids,
        "cache_position": cache_position, "causal_mask": causal_mask,
    }


def kwargs_for_layer(layer, common: dict) -> dict:
    return {
        "position_embeddings": common["position_embeddings"],
        "position_ids": common["position_ids"],
        "cache_position": common["cache_position"],
        "attention_mask": common["causal_mask"] if layer.layer_type == "full_attention" else None,
        "past_key_values": None, "use_cache": False,
    }


@torch.no_grad()
def initial_states(model, samples, device: str) -> list[torch.Tensor]:
    embedding = model.model.embed_tokens.to(device)
    try:
        return [embedding(sample.to(device)).cpu() for sample in samples]
    finally:
        model.model.embed_tokens = embedding.cpu()


@torch.no_grad()
def replay_layer(layer, states: list[torch.Tensor], common: dict, device: str) -> list[torch.Tensor]:
    kwargs = kwargs_for_layer(layer, common)
    return [layer(x.to(device), **kwargs).cpu() for x in states]


def quantizer_for(weight: torch.Tensor, bits: int, settings: ReplaySettings, name: str):
    if weight.shape[1] % settings.groupsize:
        raise ValueError(f"G{settings.groupsize} does not divide {name}'s input dimension.")
    return GPTQWeightQuantizer(
        weight.detach(), name=name, nbits=bits, groupsize=settings.groupsize,
        blocksize=settings.blocksize, percdamp=settings.percdamp, mse=True,
        actorder=False, static_groups=False,
    )


@torch.no_grad()
def dequantized_weight(quantizer):
    if quantizer.nsamples == 0:
        raise ValueError(f"No GPTQ calibration observations for {quantizer.name}.")
    q, scales, zeros = quantizer.quantize()
    return quantizer.dequantize(q, scales, zeros).reshape_as(quantizer.W)


@torch.no_grad()
def capture_routing(layer, states, common, device):
    """One baseline forward per sample; cache MLP inputs and router decisions on CPU."""
    mlp_inputs, indices, scores = [], [], []

    def input_hook(_module, args):
        mlp_inputs.append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())

    def gate_hook(_module, _args, output):
        scores.append(output[1].detach().cpu())
        indices.append(output[2].detach().cpu())

    handles = [layer.mlp.register_forward_pre_hook(input_hook), layer.mlp.gate.register_forward_hook(gate_hook)]
    try:
        outputs = replay_layer(layer, states, common, device)
    finally:
        for handle in handles:
            handle.remove()
    if len(indices) != len(states) or len(mlp_inputs) != len(states):
        raise RuntimeError("Router and MLP activation capture counts disagree.")
    return outputs, mlp_inputs, indices, scores


def selected_tokens(mlp_inputs, indices, scores, expert_idx: int, device: str):
    xs, weights = [], []
    for hidden, ids, values in zip(mlp_inputs, indices, scores):
        token_idx, slot_idx = torch.where(ids == expert_idx)
        if token_idx.numel():
            xs.append(hidden[token_idx])
            weights.append(values[token_idx, slot_idx])
    if not xs:
        return None, None
    return torch.cat(xs).to(device), torch.cat(weights).to(device)


def fallback_tokens(mlp_inputs, device: str, max_tokens: int = 2048):
    """Prevent an unvisited expert from being zeroed by GPTQ's dead-column rule."""
    parts = []
    remaining = max_tokens
    for hidden in mlp_inputs:
        part = hidden[:remaining]
        parts.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    return torch.cat(parts).to(device)


def release_layer(layer, model, layer_idx: int):
    model.model.layers[layer_idx] = layer.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
