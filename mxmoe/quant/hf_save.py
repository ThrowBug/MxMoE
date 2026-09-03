"""Hugging Face checkpoint I/O for dequantized MxMoE weights."""

import json
from pathlib import Path

import torch


@torch.no_grad()
def save_fake_quant_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    metadata: dict,
    max_shard_size: str = "5GB",
):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device="cpu", dtype=torch.bfloat16)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_path)

    with (output_path / "mxmoe_quantization_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return output_path
