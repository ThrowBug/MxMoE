import torch
import torch.nn as nn
from transformers import PreTrainedModel

@torch.no_grad()
def move_weight_to_cpu(model: PreTrainedModel, layer_idx: int = -1) -> list[dict[str, torch.Tensor]]:
    weights = [{} for _ in range(len(model.model.layers))]

    for layer_i, layer in enumerate(model.model.layers):
        for p_name, p in layer.named_parameters():
            if layer_idx != -1 and layer_idx != layer_i: continue
            weights[layer_i][p_name] = p.to("cpu")

    return weights

@torch.no_grad()
def recover_weight_from_cpu(model: PreTrainedModel, weights: list[list[torch.Tensor]], layer_idx: int = -1, exp_idx: int=-1):
    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue
        for p_name, p in layer.named_parameters():
            p.data.copy_(weights[layer_i][p_name])


def is_non_moe_layer(model_id: str, layer_idx: int):
    if model_id == "mixtral":
        return False
    elif model_id in [
        "qwen2_moe",
        "qwen2_moe_57b",
        "qwen3_moe_30b_a3b_instruct_2507",
    ]:
        return False
    elif model_id == "ds2":
        # TODO:
        return layer_idx == 0
    else:
        raise ValueError(f"Unsupported model: {model_id}")


@torch.no_grad()
def get_linear_block_weight(model_id:str, model_weight: dict|PreTrainedModel, layer_idx: int, expert_idx: int, linear_block_name: str, return_key=False) -> torch.Tensor:
    '''
    linear_block_name: [gate, up, down]
    '''
    if isinstance(model_weight, PreTrainedModel):
        model_weight = dict(model_weight.named_parameters())

    mlp_block_name = MOE_MLP_NAME_MAP[model_id]
    linear_block_name = MOE_WEIGHT_NAME_MAP[model_id][linear_block_name]

    if is_non_moe_layer(model_id, layer_idx):
        mlp_weight_key = f"{mlp_block_name}.{linear_block_name}"
    else:
        mlp_weight_key = f"{mlp_block_name}.experts.{expert_idx}.{linear_block_name}"

        # handle the shared expert
        if model_weight.get(f"model.layers.{layer_idx}.{mlp_weight_key}.weight", None) is None:
            if model_id == "ds2":
                mlp_weight_key = f"{mlp_block_name}.shared_experts.{linear_block_name}"
            elif model_id in ["qwen2_moe", "qwen2_moe_57b"]:
                mlp_weight_key = f"{mlp_block_name}.shared_expert.{linear_block_name}"
            else:
                raise ValueError(f"Unsupported model: {model_id}")
    weight_key = f"model.layers.{layer_idx}.{mlp_weight_key}.weight"

    if return_key: return weight_key, model_weight[weight_key]
    return model_weight[weight_key]


@torch.no_grad()
def offload_moe_weights(model_id:str, model: PreTrainedModel, layer_id:int=-1, expert_id:int=-1, linear_block:str=""):
    cpu_weights: dict[str, torch.Tensor] = {}
    for layer_idx in range(len(model.model.layers)):
        if layer_id != -1 and layer_id != layer_idx: continue
        for expert_idx in range(len(get_expert_linears(model, layer_idx))):
            if expert_id != -1 and expert_id != expert_idx: continue
            for linear_block_name in ["gate", "up", "down"]:
                if len(linear_block) > 0 and linear_block_name != linear_block: continue
                wname,w = get_linear_block_weight(model_id, model, layer_idx, expert_idx, linear_block_name, True)
                cpu_weights[wname] = w.cpu()

    return cpu_weights

@torch.no_grad()
def substitue_moe_weights(model_id:str, model: PreTrainedModel, cpu_weights: dict[str, torch.Tensor], layer_id:int=-1, expert_id:int=-1, linear_block:str=""):
    for layer_idx in range(len(model.model.layers)):
        if layer_id != -1 and layer_id != layer_idx: continue
        for expert_idx in range(len(get_expert_linears(model, layer_idx))):
            if expert_id != -1 and expert_id != expert_idx: continue
            for linear_block_name in ["gate", "up", "down"]:
                if len(linear_block) > 0 and linear_block_name != linear_block: continue

                old_weight = get_linear_block_weight(model_id, model, layer_idx, expert_idx, linear_block_name)
                linear_block_weight = get_linear_block_weight(model_id, cpu_weights, layer_idx, expert_idx, linear_block_name)
                old_weight.copy_(linear_block_weight)


def get_expert_linears(model: PreTrainedModel, layer_idx: int = -1, exclude_non_moe_layer: bool=False) -> list[list[nn.Module]] | list[nn.Module]:
    '''
    return: the list of experts in MoE module of each layer

    exclude_non_moe_layer: for deepseek_v2, the first layer is traditional MLP layer
    '''
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    experts = [[] for _ in range(len(model.model.layers))]
    model_type = model.config.model_type

    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue
        if model_type == "deepseek_v2":
            # non-moe layer
            if layer_i == 0 and not exclude_non_moe_layer:
                experts[layer_i].append(layer.mlp)
                continue

            for expert in layer.mlp.experts:
                experts[layer_i].append(expert)
            experts[layer_i].append(layer.mlp.shared_experts)

        elif model_type == "mixtral":
            for expert in layer.block_sparse_moe.experts:
                experts[layer_i].append(expert)
        elif model_type == "qwen2_moe":
            for expert in layer.mlp.experts:
                experts[layer_i].append(expert)
            experts[layer_i].append(layer.mlp.shared_expert)
        elif model_type == "qwen3_moe":
            for expert in layer.mlp.experts:
                experts[layer_i].append(expert)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
    if layer_idx != -1:
        experts = experts[layer_idx]
    return experts


def get_moe_gate_linears(model: PreTrainedModel, layer_idx: int = -1) -> list[nn.Module] | nn.Module:
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    expert_gates = [[] for _ in range(len(model.model.layers))]
    model_type = model.config.model_type

    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue
        if model_type == "deepseek_v2":
            # non-moe layer
            if layer_i == 0:
                continue

            gate = getattr(layer.mlp, "gate", None)
            if gate is not None: expert_gates[layer_i].append(gate.weight)
        elif model_type == "mixtral":
            gate = getattr(layer.block_sparse_moe, "gate", None)
            if gate is not None: expert_gates[layer_i].append(gate)
        elif model_type == "qwen2_moe":
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None: expert_gates[layer_i].append(gate)
            shared_expert_gate = getattr(layer.mlp, "shared_expert_gate", None)
            if shared_expert_gate is not None: expert_gates[layer_i].append(shared_expert_gate)
        elif model_type == "qwen3_moe":
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                expert_gates[layer_i].append(gate)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
    if layer_idx != -1:
        expert_gates = expert_gates[layer_idx]
    return expert_gates


def get_attn_linears(model: PreTrainedModel, layer_idx: int = -1) -> list[dict[str, nn.Module]] | dict[str, nn.Module]:
    '''
    return: the list of weights in attention module of each layer

    '''
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    weights = [{} for _ in range(len(model.model.layers))]

    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue

        for m_name, m in layer.self_attn.named_modules():
            if not isinstance(m, nn.Linear): continue
            weights[layer_i][m_name] = m

    if layer_idx != -1:
        weights = weights[layer_idx]
    return weights

MOE_WEIGHT_NAME_REMAP = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "w1": "gate",
    "w3": "up",
    "w2": "down",
}

MOE_MLP_NAME_MAP = {
    "ds2": "mlp",
    "qwen2_moe": "mlp",
    "qwen2_moe_57b": "mlp",
    "qwen3_moe_30b_a3b_instruct_2507": "mlp",
    "mixtral": "block_sparse_moe",
}
MOE_WEIGHT_NAME_MAP = {
    **dict.fromkeys([
        "ds2",
        "qwen2_moe",
        "qwen2_moe_57b",
        "qwen3_moe_30b_a3b_instruct_2507",
    ], {
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }),
    "mixtral": {
        "gate": "w1",
        "up": "w3",
        "down": "w2",
    }
}

def get_linears_in_one_expert(expert: nn.Module) -> dict[str, nn.Linear]:
    '''
    return: the list of weights in one expert
    '''
    filters = [
        "gate_proj", "up_proj", "down_proj",
        "w1", "w2", "w3", # for mixtral
    ]

    weights = {}
    for m_name, m in expert.named_modules():
        if any(k in m_name for k in filters): 
            wname = MOE_WEIGHT_NAME_REMAP[m_name]
            weights[wname] = m

    return weights


def get_device_map(model_id: str):
    device_map = "auto"
    if torch.cuda.device_count() == 2:
        if model_id == "qwen2_moe_57b":
            device_map = {
                "model.embed_tokens": "cuda:0",
                "model.rotary_emb": "cuda:0",
                **{
                    f"model.layers.{k}": 0 for k in range(0, 14)
                },
                **{
                    f"model.layers.{k}": 1 for k in range(14, 28)
                },
                "model.norm": "cuda:1",
                "lm_head": 1,
            }
        elif model_id == "mixtral":
            device_map = {
                "model.embed_tokens": "cuda:0",
                **{
                    f"model.layers.{k}": 0 for k in range(0, 16)
                },
                **{
                    f"model.layers.{k}": 1 for k in range(16, 32)
                },
                "model.norm": "cuda:1",
                "lm_head": 1,
            }
    elif torch.cuda.device_count() == 4:
        if model_id == "qwen2_moe_57b":
            device_map = {
                "model.embed_tokens": "cuda:0",
                "model.rotary_emb": "cuda:0",
                **{f"model.layers.{k}": 0 for k in range(0, 7)},
                **{f"model.layers.{k}": 1 for k in range(7, 14)},
                **{f"model.layers.{k}": 2 for k in range(14, 21)},
                **{f"model.layers.{k}": 3 for k in range(21, 28)},
                "model.norm": "cuda:3",
                "lm_head": 3,
            }
        elif model_id == "mixtral":
            device_map = {
                "model.embed_tokens": "cuda:0",
                **{f"model.layers.{k}": 0 for k in range(0, 8)},
                **{f"model.layers.{k}": 1 for k in range(8, 16)},
                **{f"model.layers.{k}": 2 for k in range(16, 24)},
                **{f"model.layers.{k}": 3 for k in range(24, 32)},
                "model.norm": "cuda:3",
                "lm_head": 3,
            }
    return device_map

def load_hf_model(model_id: str, ckpt=None, rotation=False, dtype="auto"):
    from project_config import ID2NAME
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
    except ImportError:
        Qwen3MoeDecoderLayer = None

    from accelerate import load_checkpoint_and_dispatch, init_empty_weights
    from mxmoe.quant.rotation import fuse_layer_norms


    model_name = ID2NAME[model_id]

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    if rotation:
        print("Fusing Layer norm ...")
        fuse_layer_norms(model)

    ckpt = ckpt if ckpt is not None else model_name
    no_split_module_classes = [MixtralDecoderLayer, Qwen2MoeDecoderLayer]
    if Qwen3MoeDecoderLayer is not None:
        no_split_module_classes.append(Qwen3MoeDecoderLayer)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt,
        device_map = get_device_map(model_id),
        no_split_module_classes=no_split_module_classes,
    )
    return model


def load_tokenizer(model_id: str):
    from project_config import ID2NAME
    from transformers import AutoTokenizer

    model_name = ID2NAME[model_id]
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
