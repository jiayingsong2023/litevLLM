# SPDX-License-Identifier: Apache-2.0
"""LitevLLM: Generic Base Classes for Single-GPU Models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Type, Any
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.config import VllmConfig

class LiteDecoderLayer(nn.Module):
    """
    Standardized Decoder Layer block.
    Decouples Attention and MLP components from the model-specific names.
    """
    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[Any] = None
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        # Components will be initialized by subclasses or generic factory
        self.self_attn: nn.Module = None 
        self.mlp: nn.Module = None
        self.input_layernorm: nn.Module = None
        self.post_attention_layernorm: nn.Module = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # 1. Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        # 2. MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class LiteModel(nn.Module):
    """
    Core architecture builder.
    Abstracts Embedding, Layers stack, and Final Norm.
    """
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        prefix: str = ""
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList([
            # To be populated by specific model implementations
        ])
        self.norm = None # Final norm

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        hidden_states = self.norm(hidden_states)
        return hidden_states