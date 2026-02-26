# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig

class Lfm2MoeConfig(PretrainedConfig):

    model_type = "lfm2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 7168,
        moe_intermediate_size: int = 1792,
        num_hidden_layers: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 128_000,
        use_cache: bool = True,
        norm_eps: float = 0.00001,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        conv_bias: bool = False,
        conv_L_cache: int = 3,
        num_dense_layers: int = 2,
        num_experts_per_tok: int = 4,
        num_experts: int = 32,
        use_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        rope_theta = kwargs.pop("rope_theta", 1000000.0)
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.norm_eps = norm_eps

        # attn operator config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # custom operator config
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache

        # moe config
        self.num_dense_layers = num_dense_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.use_expert_bias = use_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.layer_types = layer_types

        tie_word_embeddings = kwargs.get(
            "tie_embedding", tie_word_embeddings
        )  # to fit original config keys
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["Lfm2MoeConfig"]
