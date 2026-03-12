# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.config import VllmConfig
from .lite_config import LiteConfig

def _get_eps(config):
    return getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))

class DeepseekV2Attention(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        # Standard names for loader
        self.q_proj = LiteLinear(config.hidden_size, self.num_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj")
        self.k_proj = LiteLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.k_proj")
        self.v_proj = LiteLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.v_proj")
        self.o_proj = LiteLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")

    def forward(self, x):
        q = self.q_proj(x)
        return self.o_proj(q)

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config; eps = _get_eps(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = DeepseekV2Attention(config, quant_config, prefix=prefix)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        # Standard GGUF Mapping for MLP
        self.mlp = type('obj', (object,), {
            'gate_proj': LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj"),
            'up_proj': LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj"),
            'down_proj': LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
        })

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        if hasattr(self, "_fast_forward") and self._fast_forward is not None:
            return self._fast_forward(hidden_states, positions, kv_cache, attn_metadata)
        h = self.input_layernorm(hidden_states); attn_out = self.self_attn(h)
        hidden_states = hidden_states + attn_out; h = self.post_attention_layernorm(hidden_states)
        gate = self.mlp.gate_proj(h); up = self.mlp.up_proj(h)
        return hidden_states + self.mlp.down_proj(F.silu(gate) * up)

    def compile_fast_dispatch(self):
        self._fast_forward = None

class DeepseekV2Model(nn.Module):
    def __init__(self, hf_config, quant_config, prefix="model"):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(self.config, quant_config, f"{prefix}.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.norm(x)

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix="model"):
        super().__init__()
        self.model = DeepseekV2Model(vllm_config.model_config.hf_config, vllm_config.quant_config, "model")
        self.lm_head = LiteLinear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states[:, -1:, :])

class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): pass
