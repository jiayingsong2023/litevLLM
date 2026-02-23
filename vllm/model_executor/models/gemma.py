# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Gemma model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Tuple, Any
from transformers import GemmaConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.lite_base import LiteForCausalLM, LiteModel, LiteDecoderLayer

class GemmaLiteDecoderLayer(LiteDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", layer_idx: int = 0):
        super().__init__(vllm_config, prefix, layer_idx)
        # Gemma uses GemmaRMSNorm
        self.input_layernorm = GemmaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

class GemmaLiteModel(LiteModel):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config, prefix)
        # Override layers with Gemma-specific version
        self.layers = nn.ModuleList([
            GemmaLiteDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}", layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Scaling factor for embeddings
        self.normalizer = self.config.hidden_size**0.5

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        
        x *= self.normalizer
        
        for layer in self.layers:
            x = layer(positions, x, **kwargs)
        return self.norm(x)

class GemmaForCausalLM(LiteForCausalLM):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config, prefix)
        # Re-init model with GemmaLiteModel
        self.model = GemmaLiteModel(vllm_config, prefix=f"{prefix}.model")
        
    def compute_logits(self, hidden_states):
        # Gemma ties embeddings by default
        return self.logits_processor(self.model.embed_tokens, hidden_states)