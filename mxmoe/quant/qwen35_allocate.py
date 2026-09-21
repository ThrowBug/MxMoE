"""Accuracy-only exact-budget allocation for Qwen3.5 packed routed experts.

Gate and up share one bit choice; down has an independent choice.  A dynamic
program avoids Gurobi's small-license variable limit at 256 experts/layer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mxmoe.kernels.qconfig import QLinearConfig
from mxmoe.quant.artifact_utils import load_optional_metadata, validate_calibration_artifacts, write_metadata

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
MODEL_ID = "qwen3_5_moe_35b_a3b"
def validate_run_metadata(metadata, model):
    """Require trace and loss files to belong to the same model and bit setting."""
    fixed = metadata[0][1].get("fixed_config")
    if not isinstance(fixed, dict) or any(
        item.get("model") != model or item.get("fixed_config") != fixed
        for _, item in metadata
    ):
        raise ValueError(
            "Trace and loss artifacts must use the same model and fixed-bit settings."
        )
    return fixed


def allocate_layer(losses: dict[int, list[list[float]]], nominal_bits: float):
    experts = len(next(iter(losses.values())))
    if any(len(losses[bit]) != experts for bit in (1, 2, 3)):
        raise ValueError("Loss files disagree on the routed-expert count.")
    target = nominal_bits * experts * 3
    if not math.isclose(target, round(target), abs_tol=1e-8):
        raise ValueError("Nominal target cannot be represented with integer bit choices.")
    extra = round(target) - experts * 3
    if not 0 <= extra <= experts * 6:
        raise ValueError("Nominal bit target must lie between 1 and 3.")
    options = [(gate, down, 2 * (gate - 1) + (down - 1))
               for gate in (1, 2, 3) for down in (1, 2, 3)]
    dp = [math.inf] * (extra + 1)
    dp[0] = 0.0
    parents = []
    for expert in range(experts):
        updated = [math.inf] * (extra + 1)
        choices = [None] * (extra + 1)
        for used, prior in enumerate(dp):
            if not math.isfinite(prior):
                continue
            for gate, down, cost in options:
                new_used = used + cost
                if new_used > extra:
                    continue
                value = prior + sum(losses[gate][expert][:2]) + losses[down][expert][2]
                if value < updated[new_used]:
                    updated[new_used] = value
                    choices[new_used] = (used, gate, down)
        dp = updated
        parents.append(choices)
    if not math.isfinite(dp[extra]):
        raise ValueError("No allocation satisfies the requested nominal bit target.")
    selected = [None] * experts
    used = extra
    for expert in reversed(range(experts)):
        parent = parents[expert][used]
        if parent is None:
            raise RuntimeError("Broken allocation backtrace.")
        used, gate, down = parent
        selected[expert] = (gate, down)
    return selected, dp[extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--loss_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nominal_bits", type=float, default=2.0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite allocation {output}.")
    with Path(args.trace).open(encoding="utf-8") as stream:
        trace = json.load(stream)
    trace_meta = load_optional_metadata(args.trace, artifact_name="Qwen3.5 trace")
    if trace_meta is None:
        raise ValueError("Qwen3.5 trace requires a metadata sidecar.")
    expected = {"dataset": "c4", "nsamples": args.nsamples,
                "seqlen": args.seqlen, "seed": args.seed}
    losses = {}
    metadata = [("trace", trace_meta)]
    for bit in (1, 2, 3):
        path = Path(args.loss_dir) / (
            f"{MODEL_ID}-MOE-gptq-W{bit}A16_g128_asym-c4-"
            f"{args.nsamples}-{args.seqlen}-layer_out_norm.json"
        )
        with path.open(encoding="utf-8") as stream:
            losses[bit] = json.load(stream)
        sidecar = load_optional_metadata(path, artifact_name=f"Qwen3.5 W{bit} loss")
        if sidecar is None:
            raise ValueError("Qwen3.5 loss requires a metadata sidecar.")
        metadata.append((f"loss[{bit}]", sidecar))
    token_hash = validate_calibration_artifacts(metadata, expected)
    if token_hash is None:
        raise ValueError("Qwen3.5 calibration token hash is required.")
    fixed = validate_run_metadata(metadata, args.model)
    num_layers = trace["num_layers"]
    num_experts = len(trace["layer-0"]["access_freq"])
    result, total_loss = {}, 0.0
    for layer_idx in range(num_layers):
        layer = str(layer_idx)
        if len(trace[f"layer-{layer_idx}"]["access_freq"]) != num_experts:
            raise ValueError(f"Layer {layer_idx} trace expert count disagrees.")
        values = {}
        for bit in (1, 2, 3):
            if set(losses[bit][layer]) != {str(e) for e in range(num_experts)}:
                raise ValueError(f"Layer {layer_idx} W{bit} loss expert IDs disagree.")
            values[bit] = [losses[bit][layer][str(e)] for e in range(num_experts)]
            if any(len(item) != 3 or not all(math.isfinite(float(x)) and float(x) >= 0 for x in item)
                   for item in values[bit]):
                raise ValueError(f"Layer {layer_idx} W{bit} has invalid losses.")
        choices, objective = allocate_layer(values, args.nominal_bits)
        total_loss += objective
        experts = {}
        for expert_idx, (gate_bit, down_bit) in enumerate(choices):
            experts[str(expert_idx)] = {
                "gate": QLinearConfig(w_bits=gate_bit, w_gsize=128, a_gsize=128).to_dict(),
                "up": QLinearConfig(w_bits=gate_bit, w_gsize=128, a_gsize=128).to_dict(),
                "down": QLinearConfig(w_bits=down_bit, w_gsize=128, a_gsize=128).to_dict(),
            }
        result[layer] = {"experts": experts}
        print(f"Allocated layer {layer_idx + 1}/{num_layers}: loss={objective:.6g}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream)
    write_metadata(output, {
        "format": "mxmoe-allocation-v1", "model": args.model, "model_id": MODEL_ID,
        "qtype": "gptq", "calibration": {**expected, "input_ids_sha256": token_hash},
        "fixed_config": fixed, "nominal_bits": args.nominal_bits,
        "routed_experts_only": True, "objective": "accuracy-only",
        "total_loss": total_loss, "trace_file": str(Path(args.trace).resolve()),
    })


if __name__ == "__main__":
    main()
