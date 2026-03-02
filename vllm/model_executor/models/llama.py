# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.models.lite_base import LiteModel, LiteDecoderLayer
from vllm.attention.backends.triton_attn import TritonAttention

class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False,
                                     quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False,
                                   quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = Silu()

    def forward(self, x, lora_mapping=None):
        gate_up = self.gate_up_proj(x, lora_mapping=lora_mapping)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up, lora_mapping=lora_mapping)

class LlamaLayer(LiteDecoderLayer):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.self_attn = TritonAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = LlamaMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, lora_mapping=lora_mapping)
        return residual + hidden_states

class LlamaModel(LiteModel):
    def __init__(self, vllm_config, prefix=""):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer_kv_cache = kv_caches[i]
            hidden_states = self.layers[i](hidden_states, positions, layer_kv_cache, attn_metadata, lora_mapping=lora_mapping)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = LlamaModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size,
                                 vllm_config.model_config.hf_config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        return self.lm_head(hidden_states, lora_mapping=lora_mapping)
