"""Create a Qwen3-30B-A3B-Instruct-2507 MxMoE fake-quant checkpoint."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mxmoe.kernels.qconfig import build_qmodel_cfg_from_json
from mxmoe.quant.data_utils import get_calibration_samples
from mxmoe.quant.artifact_utils import (
    load_optional_metadata,
    validate_calibration_artifacts,
)
from mxmoe.quant.hf_save import save_fake_quant_checkpoint
from mxmoe.quant.mixed_gptq import GPTQConfig, quantize_model_mixed_gptq


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MxMoE mixed-bit GPTQ for Qwen3-30B-A3B-Instruct-2507."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--qconfig", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib_dataset", default="c4", choices=["c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--attn_wbits", type=int, default=4)
    parser.add_argument("--dense_wbits", type=int, default=4)
    parser.add_argument("--expected_nominal_bits", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--allow_artifact_mismatch",
        action="store_true",
        help="Allow a qconfig sidecar that disagrees with this quantization run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Qwen3 GPTQ quantization requires a CUDA device.")
    set_seed(args.seed)

    allocation_metadata = load_optional_metadata(
        args.qconfig, artifact_name="allocation qconfig"
    )
    if allocation_metadata is not None:
        try:
            validate_calibration_artifacts(
                [("allocation qconfig", allocation_metadata)],
                {
                    "dataset": args.calib_dataset,
                    "nsamples": args.nsamples,
                    "seqlen": args.seqlen,
                    "seed": args.seed,
                },
            )
            expected_model = allocation_metadata.get("model")
            if expected_model is not None and expected_model != args.model:
                raise ValueError(
                    f"Allocation model mismatch: expected {expected_model!r}, "
                    f"got {args.model!r}."
                )
            if allocation_metadata.get("qtype") not in (None, "gptq"):
                raise ValueError(
                    "Qwen3 quantization expects a plain GPTQ allocation, got "
                    f"{allocation_metadata['qtype']!r}."
                )
        except ValueError as error:
            if not args.allow_artifact_mismatch:
                raise
            warnings.warn(f"Ignoring artifact mismatch: {error}", stacklevel=1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    samples, calibration_metadata = get_calibration_samples(
        tokenizer,
        calib_dataset=args.calib_dataset,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    if allocation_metadata is not None:
        try:
            validate_calibration_artifacts(
                [
                    ("allocation qconfig", allocation_metadata),
                    ("quantization inputs", {"calibration": calibration_metadata}),
                ],
                calibration_metadata,
            )
        except ValueError as error:
            if not args.allow_artifact_mismatch:
                raise
            warnings.warn(f"Ignoring artifact mismatch: {error}", stacklevel=1)
    qconfig = build_qmodel_cfg_from_json(args.qconfig)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.seqlen = args.seqlen

    gptq_config = GPTQConfig(
        groupsize=args.groupsize,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        mse=True,
        actorder=False,
        static_groups=False,
        attn_wbits=args.attn_wbits,
        dense_wbits=args.dense_wbits,
    )
    report = quantize_model_mixed_gptq(
        model, samples, qconfig, gptq_config, device=args.device
    )
    nominal_average = report["allocation"]["nominal_average_bits"]
    if abs(nominal_average - args.expected_nominal_bits) > 1e-6:
        raise ValueError(
            f"Allocation nominal average is {nominal_average:.8f}, expected "
            f"{args.expected_nominal_bits:.8f}."
        )

    metadata = {
        "format": "mxmoe-dequantized-bfloat16",
        "model": args.model,
        "qconfig": str(Path(args.qconfig).resolve()),
        "calibration": calibration_metadata,
        **report,
    }
    output_path = save_fake_quant_checkpoint(
        model,
        tokenizer,
        args.output,
        metadata,
        max_shard_size=args.max_shard_size,
    )
    print(json.dumps(metadata["allocation"], indent=2))
    print(f"Saved MxMoE fake-quant checkpoint to {output_path}")


if __name__ == "__main__":
    main()
