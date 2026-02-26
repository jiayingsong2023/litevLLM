# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

import torch
from huggingface_hub import constants
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.model_arch import (
    ModelArchitectureConfig,
)
from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    ConfigFormat,
    try_get_safetensors_metadata,
)
from vllm.utils.torch_utils import common_broadcastable_dtype

logger = init_logger(__name__)

@contextmanager
def _maybe_patch_hf_hub_constants(config_format: ConfigFormat) -> Iterator[None]:
    if config_format == "mistral":
        hf_safetensors_single_file = constants.SAFETENSORS_SINGLE_FILE
        hf_safetensors_index_file = constants.SAFETENSORS_INDEX_FILE
        constants.SAFETENSORS_SINGLE_FILE = "consolidated.safetensors"
        constants.SAFETENSORS_INDEX_FILE = "consolidated.safetensors.index.json"
        try:
            yield
        finally:
            constants.SAFETENSORS_SINGLE_FILE = hf_safetensors_single_file
            constants.SAFETENSORS_INDEX_FILE = hf_safetensors_index_file
    else:
        yield

class ModelArchConfigConvertorBase:
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        self.hf_config = hf_config
        self.hf_text_config = hf_text_config

    def get_architectures(self) -> list[str]:
        return getattr(self.hf_config, "architectures", [])

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_hidden_layers", 0)

    def get_total_num_attention_heads(self) -> int:
        return getattr(self.hf_text_config, "num_attention_heads", 0)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            if not envs.VLLM_MLA_DISABLE:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    def get_total_num_kv_heads(self) -> int:
        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        default_factory = lambda: self.hf_text_config.num_attention_heads
        return getattr_iter(
            self.hf_text_config, attributes, default_factory=default_factory
        )

    def get_num_experts_from_block_configs(self) -> int:
        max_experts = 0
        block_configs = getattr(self.hf_text_config, "block_configs", None)
        if block_configs:
            for block in block_configs:
                if isinstance(block, dict):
                    if block.get("block_type", "") == "moe":
                        max_experts = max(max_experts, block.get("n_routed_experts", 0))
                else:
                    if getattr(block, "block_type", "") == "moe":
                        max_experts = max(
                            max_experts, getattr(block, "n_routed_experts", 0)
                        )
        return max_experts

    def get_num_experts(self) -> int:
