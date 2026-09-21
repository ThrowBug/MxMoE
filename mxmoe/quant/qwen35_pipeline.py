"""Qwen3.5-35B-A3B text-only MxMoE calibration and fake-quant workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mxmoe.kernels.qconfig import build_qmodel_cfg_from_json
from mxmoe.quant.artifact_utils import load_optional_metadata, validate_calibration_artifacts, write_metadata
from mxmoe.quant.data_utils import get_calibration_samples
from mxmoe.quant.hf_save import save_fake_quant_checkpoint
from mxmoe.quant.qwen35_core import (
    FIXED_KEYS, ReplaySettings, capture_routing, check_fixed_bits, check_text_model,
    dequantized_weight, expert_weight, fallback_tokens, fixed_linears, initial_states,
    quantizer_for, release_layer, replay_kwargs, replay_layer, selected_tokens,
    validate_allocation,
)


DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
MODEL_ID = "qwen3_5_moe_35b_a3b"


def fixed_settings(args):
    result = {key: getattr(args, key) for key in FIXED_KEYS}
    check_fixed_bits(result)
    return result


def load_model(path, attn_impl="eager"):
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cpu", dtype=torch.bfloat16, attn_implementation=attn_impl
    )
    model.eval()
    check_text_model(model)
    model.config.use_cache = False
    return model


def calibration(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    samples, metadata = get_calibration_samples(
        tokenizer, calib_dataset="c4", nsamples=args.nsamples,
        seqlen=args.seqlen, seed=args.seed,
    )
    return tokenizer, samples, metadata


def _start(model, samples, args):
    states = initial_states(model, samples, args.device)
    common = replay_kwargs(model, args.seqlen, args.device)
    return states, common


@torch.no_grad()
def trace(args, samples, calib_meta):
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite trace {args.output}.")
    model = load_model(args.model)
    states, common = _start(model, samples, args)
    num_experts = model.config.num_experts
    topk = model.config.num_experts_per_tok
    result = {
        "topk": topk, "NK": [model.config.moe_intermediate_size, model.config.hidden_size],
        "num_layers": len(model.model.layers), "num_tokens": args.seqlen,
        "num_samples": len(samples), "num_shared_experts": 1,
        "model": args.model, "calibration": calib_meta,
        "fixed_config": fixed_settings(args),
    }
    for layer_idx, original in enumerate(model.model.layers):
        layer = original.to(args.device)
        states, _hidden, indices, scores = capture_routing(layer, states, common, args.device)
        freq = torch.zeros(num_experts, dtype=torch.float64)
        weight_sum = torch.zeros(num_experts, dtype=torch.float64)
        for ids, values in zip(indices, scores):
            freq.scatter_add_(0, ids.flatten(), torch.ones(ids.numel(), dtype=torch.float64))
            weight_sum.scatter_add_(0, ids.flatten(), values.flatten().double())
        freq = (freq / len(samples)).round().long()
        weight_sum = weight_sum / len(samples)
        percentiles = {}
        for count in (1, 8, 16, 32, 64, 96):
            count = min(count, num_experts)
            ids = freq.topk(count).indices
            percentiles[count] = {
                "topk_ids": ids.tolist(), "freq": freq[ids].tolist(),
                "percent": freq[ids].sum().item() / max(freq.sum().item(), 1),
            }
        result[f"layer-{layer_idx}"] = {
            "access_freq": freq.tolist(), "weights_sum": weight_sum.tolist(),
            "percentile_stats": percentiles,
        }
        del _hidden, indices, scores
        release_layer(layer, model, layer_idx)
        print(f"Trace: layer {layer_idx + 1}/{len(model.model.layers)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream)
    write_metadata(output, {"model": args.model, "calibration": calib_meta,
                            "fixed_config": fixed_settings(args)})


def _expert_activations(experts, expert_idx, xs):
    gate = F.linear(xs, expert_weight(experts, expert_idx, "gate"))
    up = F.linear(xs, expert_weight(experts, expert_idx, "up"))
    return gate, up, experts.act_fn(gate) * up


@torch.no_grad()
def quantize_expert(experts, expert_idx, xs, bits_by_name, settings, *, score=None):
    """Quantize logical slices, optionally return exact isolated layer-output errors."""
    gate, up, intermediate = _expert_activations(experts, expert_idx, xs)
    baseline_out = F.linear(intermediate, expert_weight(experts, expert_idx, "down")) if score is not None else None
    result = {}
    for name in ("gate", "up", "down"):
        weight = expert_weight(experts, expert_idx, name)
        bits = bits_by_name[name]
        q = quantizer_for(weight, bits, settings, f"expert.{expert_idx}.{name}")
        q.add_batch(intermediate if name == "down" else xs)
        restored = dequantized_weight(q)
        if score is not None:
            if name == "gate":
                candidate = F.linear(experts.act_fn(F.linear(xs, restored)) * up,
                                     expert_weight(experts, expert_idx, "down"))
            elif name == "up":
                candidate = F.linear(experts.act_fn(gate) * F.linear(xs, restored),
                                     expert_weight(experts, expert_idx, "down"))
            else:
                candidate = F.linear(intermediate, restored)
            result[name] = torch.linalg.vector_norm(
                ((candidate.float() - baseline_out.float()) * score[:, None].float()).flatten()
            ).item()
        else:
            weight.copy_(restored)
            result[name] = weight.numel()
    return result


@torch.no_grad()
def collect_loss(args, samples, calib_meta):
    paths = {
        bit: Path(args.output_dir) / (
            f"{MODEL_ID}-MOE-gptq-W{bit}A16_g128_asym-c4-"
            f"{args.nsamples}-{args.seqlen}-layer_out_norm.json"
        ) for bit in (1, 2, 3)
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite collected loss {path}.")
    model = load_model(args.model)
    states, common = _start(model, samples, args)
    settings = ReplaySettings(args.device, args.groupsize, args.blocksize, args.percdamp)
    all_losses = {str(bit): {} for bit in (1, 2, 3)}
    for layer_idx, original in enumerate(model.model.layers):
        layer = original.to(args.device)
        next_states, hidden, indices, scores = capture_routing(layer, states, common, args.device)
        experts = layer.mlp.experts
        for bit in (1, 2, 3):
            all_losses[str(bit)][str(layer_idx)] = {}
        for expert_idx in range(experts.gate_up_proj.shape[0]):
            xs, routing_weight = selected_tokens(hidden, indices, scores, expert_idx, args.device)
            if xs is None:
                for bit in (1, 2, 3):
                    all_losses[str(bit)][str(layer_idx)][str(expert_idx)] = [0.0, 0.0, 0.0]
                continue
            for bit in (1, 2, 3):
                errors = quantize_expert(
                    experts, expert_idx, xs, {name: bit for name in ("gate", "up", "down")},
                    settings, score=routing_weight,
                )
                all_losses[str(bit)][str(layer_idx)][str(expert_idx)] = [
                    errors[name] for name in ("gate", "up", "down")
                ]
        states = next_states
        del hidden, indices, scores, xs, routing_weight
        release_layer(layer, model, layer_idx)
        print(f"Collect loss: layer {layer_idx + 1}/{len(model.model.layers)}", flush=True)
        for bit in (1, 2, 3):
            output = paths[bit]
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(f"{output}.partial")
            with partial.open("w", encoding="utf-8") as stream:
                json.dump(all_losses[str(bit)], stream)
            if layer_idx + 1 == len(model.model.layers):
                partial.replace(output)
                write_metadata(output, {
                    "model": args.model, "qtype": "gptq", "calibration": calib_meta,
                    "fixed_config": fixed_settings(args),
                    "loss_definition": "isolated routed-expert layer-output L2 norm",
                })


@torch.no_grad()
def quantize_final(args, tokenizer, samples, calib_meta):
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite quantized checkpoint {args.output}.")
    allocation_meta = load_optional_metadata(args.qconfig, artifact_name="Qwen3.5 allocation")
    if allocation_meta is None:
        raise ValueError("Qwen3.5 allocation requires a metadata sidecar.")
    validate_calibration_artifacts(
        [("allocation", allocation_meta), ("current inputs", {"calibration": calib_meta})], calib_meta
    )
    if (allocation_meta.get("model") != args.model
            or allocation_meta.get("fixed_config") != fixed_settings(args)):
        raise ValueError("Allocation model or fixed-bit settings disagree.")
    model = load_model(args.model)
    qconfig = build_qmodel_cfg_from_json(args.qconfig)
    summary = validate_allocation(model, qconfig, args.groupsize)
    if abs(summary["nominal_average_bits"] - args.expected_nominal_bits) > 1e-6:
        raise ValueError("Routed-expert nominal average does not match the requested target.")
    states, common = _start(model, samples, args)
    settings = ReplaySettings(args.device, args.groupsize, args.blocksize, args.percdamp)
    report = []
    fixed_report = []
    for layer_idx, original in enumerate(model.model.layers):
        layer = original.to(args.device)
        selected = fixed_linears(layer, fixed_settings(args))
        quantizers = {}
        handles = []
        for name, (linear, bits) in selected.items():
            if bits == 16:
                continue
            q = quantizer_for(linear.weight, bits, settings, f"model.layers.{layer_idx}.{name}")
            quantizers[name] = q
            handles.append(linear.register_forward_hook(
                lambda _module, inputs, _output, q=q: q.add_batch(inputs[0].detach())
            ))
        try:
            _old_outputs, hidden, indices, scores = capture_routing(layer, states, common, args.device)
        finally:
            for handle in handles:
                handle.remove()
        del _old_outputs
        # All Hessians and routed-expert inputs above come from the same original
        # layer at X_l. Only now change weights, then advance X_l once below.
        for name, q in quantizers.items():
            linear = selected[name][0]
            linear.weight.copy_(dequantized_weight(q))
            fixed_report.append({"name": q.name, "bits": q.nbits,
                                 "parameters": linear.weight.numel()})
        # Free the fixed-linear Hessians before calibrating 256 packed experts.
        q = None
        del quantizers, handles, selected
        experts = layer.mlp.experts
        for expert_idx in range(experts.gate_up_proj.shape[0]):
            xs, _ = selected_tokens(hidden, indices, scores, expert_idx, args.device)
            unvisited = xs is None
            if unvisited:
                xs = fallback_tokens(hidden, args.device)
            cfg = qconfig.layers[str(layer_idx)].experts[str(expert_idx)]
            bits = {name: linear.w_bits for name, linear in cfg.qmap().items()}
            quantize_expert(experts, expert_idx, xs, bits, settings)
            report.append({"layer": layer_idx, "expert": expert_idx,
                           "bits": bits, "unvisited_fallback": unvisited})
        states = replay_layer(layer, states, common, args.device)
        del hidden, indices, scores, xs
        release_layer(layer, model, layer_idx)
        print(f"Final quantization: layer {layer_idx + 1}/{len(model.model.layers)}", flush=True)
    metadata = {
        "format": "mxmoe-dequantized-bfloat16", "stage": "qwen35-mixed-gptq",
        "model": args.model, "model_id": MODEL_ID, "calibration": calib_meta,
        "fixed_config": fixed_settings(args),
        "allocation": summary, "qconfig": str(Path(args.qconfig).resolve()),
        "fixed_report": fixed_report, "expert_report": report,
    }
    save_fake_quant_checkpoint(model, tokenizer, args.output, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["trace", "collect_loss", "quantize"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    parser.add_argument("--output_dir")
    parser.add_argument("--qconfig")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_wbits", type=int, default=4)
    parser.add_argument("--linear_attn_wbits", type=int, default=4)
    parser.add_argument("--shared_expert_wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--expected_nominal_bits", type=float, default=2.0)
    args = parser.parse_args()
    check_fixed_bits(fixed_settings(args))
    if args.groupsize != 128 or args.blocksize != 128 or args.percdamp != 0.01:
        parser.error("The comparison recipe requires G128, blocksize 128, percdamp 0.01")
    if args.nsamples <= 0 or args.seqlen <= 0:
        parser.error("--nsamples and --seqlen must be positive")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Qwen3.5 GPTQ pipeline requires a CUDA device.")
    if args.stage in ("trace", "quantize") and not args.output:
        parser.error("--output is required")
    if args.stage == "collect_loss" and not args.output_dir:
        parser.error("--output_dir is required")
    if args.stage == "quantize" and not args.qconfig:
        parser.error("--qconfig is required")
    set_seed(args.seed)
    tokenizer, samples, calib_meta = calibration(args)
    if args.stage == "trace":
        trace(args, samples, calib_meta)
    elif args.stage == "collect_loss":
        collect_loss(args, samples, calib_meta)
    else:
        quantize_final(args, tokenizer, samples, calib_meta)


if __name__ == "__main__":
    main()
