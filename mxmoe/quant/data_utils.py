import hashlib
import itertools
import os
import random
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    PreTrainedModel,
    default_data_collator,
)
from transformers.testing_utils import CaptureLogger

from project_config import *


MXMOE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_C4_PATH = MXMOE_ROOT / "data" / "c4-train.00000-of-01024.json"


def set_seed(seed: int):
    """Match GEMQ's calibration seeding behavior."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def build_calib_loader(
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    max_block_size: int,
    n_blocks_for_stat: int,
    batch_size: int,
    num_workers: int,
    seed: int = 41,
):
    """Build calibration blocks with the same C4 pipeline used by GEMQ.

    The source code is intentionally kept local to MxMoE so the repository has
    no runtime dependency on GEMQ.  The C4 JSON file is expected under
    ``MxMoE/data`` when no explicit path is added in the future.
    """
    if dataset != "c4":
        raise NotImplementedError(
            f"Calibration dataset {dataset!r} is not supported by the shared GEMQ-style loader."
        )
    if not DEFAULT_C4_PATH.is_file():
        raise FileNotFoundError(
            "C4 calibration file not found. Copy GEMQ/data/"
            f"c4-train.00000-of-01024.json to {DEFAULT_C4_PATH}."
        )

    all_set = load_dataset("json", data_files={"train": str(DEFAULT_C4_PATH)})

    block_size = tokenizer.model_max_length
    if block_size > max_block_size:
        print(
            "The chosen tokenizer supports a model_max_length longer than "
            f"max_block_size={max_block_size}; using max_block_size."
        )
        block_size = max_block_size

    if n_blocks_for_stat > 0:
        calib_set = all_set["train"].shuffle(seed=seed).select(
            range(min(n_blocks_for_stat * 16, len(all_set["train"])))
        )
    else:
        print("n_blocks_for_stat <= 0, using the whole dataset.")
        calib_set = all_set["train"].shuffle(seed=seed)

    text_column_name = (
        "text" if "text" in calib_set.features else list(calib_set.features)[0]
    )
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as captured:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in captured.out:
            tok_logger.warning(
                "The long calibration examples will be chunked into fixed-size blocks."
            )
        return output

    tokenized_calib_set = calib_set.map(
        tokenize_function,
        batched=True,
        remove_columns=list(calib_set.features),
    )

    def group_texts(examples):
        concatenated_examples = {
            key: list(itertools.chain(*examples[key])) for key in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            key: [
                values[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_calib_set = tokenized_calib_set.map(group_texts, batched=True)
    if n_blocks_for_stat > 0:
        if len(lm_calib_set) <= n_blocks_for_stat:
            raise ValueError(
                f"C4 produced only {len(lm_calib_set)} blocks, but "
                f"{n_blocks_for_stat} are required."
            )
        lm_calib_set = lm_calib_set.select(range(n_blocks_for_stat))

    return DataLoader(
        lm_calib_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_data_collator,
    )


def calibration_input_ids_sha256(samples: list[torch.Tensor]) -> str:
    """Hash calibration token IDs independently of their in-memory dtype."""
    digest = hashlib.sha256()
    for sample in samples:
        normalized = sample.detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def get_calibration_samples(
    tokenizer: PreTrainedTokenizer,
    calib_dataset: str = "c4",
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
) -> tuple[list[torch.Tensor], dict]:
    """Return one ``(1, seqlen)`` tensor per calibration sample.

    Keeping this normalized representation prevents the GPTQ and routing-stat
    paths from interpreting DataLoader batches differently.
    """
    if calib_dataset != "c4":
        raise NotImplementedError(
            "Only calib_dataset='c4' is enabled for the Qwen3 comparison."
        )
    loader = build_calib_loader(
        calib_dataset,
        tokenizer,
        seqlen,
        nsamples,
        batch_size,
        num_workers,
        seed=seed,
    )
    samples = []
    for batch in loader:
        for row in batch["input_ids"]:
            samples.append(row.unsqueeze(0))
    samples = samples[:nsamples]
    if len(samples) != nsamples:
        raise ValueError(f"Expected {nsamples} calibration blocks, got {len(samples)}.")
    metadata = {
        "dataset": calib_dataset,
        "source": str(DEFAULT_C4_PATH),
        "nsamples": nsamples,
        "seqlen": seqlen,
        "seed": seed,
        "input_ids_sha256": calibration_input_ids_sha256(samples),
    }
    return samples, metadata

# def get_tokenizer(model: PreTrainedModel):
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     return tokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer:PreTrainedTokenizer, model_id: str, test_only=False):
    ID2CACHE = {
        "train":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-wiki2-train.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-wiki2-train.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-wiki2-train.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-wiki2-train.pkl",
        },
        "test":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-wiki2-test.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-wiki2-test.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-wiki2-test.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-wiki2-test.pkl",
        }
    }

    print("Loading wikitext2 ...")

    testenc_cache = ID2CACHE["test"][model_id]
    if not os.path.exists(testenc_cache):
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')["input_ids"]
        os.makedirs(os.path.dirname(testenc_cache), exist_ok=True)
        with open(testenc_cache, "wb") as f:
            pickle.dump(testenc, f)
    else:
        with open(testenc_cache, "rb") as f:
            testenc = pickle.load(f)

    # early return
    if test_only:
        return None, testenc

    trainenc_cache = ID2CACHE["train"][model_id]
    if not os.path.exists(trainenc_cache):
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        os.makedirs(os.path.dirname(trainenc_cache), exist_ok=True)
        with open(trainenc_cache, "wb") as f:
            pickle.dump(trainenc, f)
    else:
        with open(trainenc_cache, "rb") as f:
            trainenc = pickle.load(f)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # tar = inp.clone()
        # tar[:, :-1] = -100
        # trainloader.append((inp, tar))
        trainloader.append(inp)
    return trainloader, testenc

def get_humaneval_x(nsamples, seed, seqlen, tokenizer:PreTrainedTokenizer, model_id: str):
    print("Loading HumanEval-X ...")

    ID2CACHE = {
        "train":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-humanevalx-train.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-humanevalx-train.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-humanevalx-train.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-humanevalx-train.pkl",
        },
    }


    trainenc_cache = ID2CACHE["train"][model_id]
    if not os.path.exists(trainenc_cache):
        prompts = []
        for code_type in ["python", "js", "cpp", "java", "go"]:
            data = load_dataset(f"THUDM/humaneval-x", code_type)
            prompts.extend(data["test"]["prompt"])

        trainenc = tokenizer("\n\n".join(prompts), return_tensors='pt')

        os.makedirs(os.path.dirname(trainenc_cache), exist_ok=True)
        with open(trainenc_cache, "wb") as f:
            pickle.dump(trainenc, f)
    else:
        with open(trainenc_cache, "rb") as f:
            trainenc = pickle.load(f)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader, None
