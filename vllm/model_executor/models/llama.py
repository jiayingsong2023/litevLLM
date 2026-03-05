# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu

class LlamaAttention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.qkv_proj = LiteLinear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.qkv_proj")
        self.o_proj = LiteLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, lora_mapping=lora_mapping)
        q, k, v = qkv.split([self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        
        from vllm.attention.ops.triton_paged_attn import triton_paged_attention
        output = triton_paged_attention(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            kv_cache, attn_metadata["slot_mapping"], attn_metadata["seq_lens"], None, self.scale
        )
        return self.o_proj(output.view(bsz, seqlen, -1), lora_mapping=lora_mapping)

class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        g, u = self.gate_up_proj(x, lora_mapping=lora_mapping).chunk(2, dim=-1)
        return self.down_proj(self.act(g) * u, lora_mapping=lora_mapping)

class LlamaLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_id, quant_config, prefix)
        self.mlp = LlamaMLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping)
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"model.layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = LlamaModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        return self.lm_head(self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping), lora_mapping=lora_mapping)
