# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Siglip vision model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
from typing import Optional, List
from transformers import SiglipVisionConfig

from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig, quant_config=None, prefix=""):
        super().__init__()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = LiteLinear(config.hidden_size, config.intermediate_size, quant_config=quant_config, prefix=f"{prefix}.fc1")
        self.fc2 = LiteLinear(config.intermediate_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.fc2")

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig, quant_config=None, prefix=""):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = LiteLinear(self.embed_dim, 3 * self.embed_dim, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.out_proj = LiteLinear(self.embed_dim, self.embed_dim, quant_config=quant_config, prefix=f"{prefix}.out_proj")
        self.attn = Attention(self.num_heads, self.head_dim, self.scale, self.num_heads, quant_config=quant_config)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return self.out_proj(self.attn(q, k, v, None, None))

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig, quant_config=None, prefix=""):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config, quant_config, prefix=f"{prefix}.self_attn")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.embeddings = nn.Parameter(torch.randn(1, (config.image_size // config.patch_size)**2, config.hidden_size))
        self.layers = nn.ModuleList([SiglipEncoderLayer(config, quant_config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        # Simplified embedding and encoder pass
        x = self.embeddings.expand(pixel_values.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)