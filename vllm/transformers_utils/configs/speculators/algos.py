# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}

def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator

@register_speculator("eagle3")
def update_eagle3(config_dict: dict, vllm_config: dict) -> None:

    vllm_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        vllm_config["target_hidden_size"] = config_dict["target_hidden_size"]
    vllm_config["norm_before_residual"] = config_dict.get("norm_before_residual", True)
    vllm_config["architectures"] = ["Eagle3LlamaForCausalLM"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        vllm_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]
