# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe

class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # Fast path: assumes combined qkv_proj for 9B/35B
        self.qkv_proj = LiteLinear(config.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.attn_qkv")
        self.o_proj = LiteLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_output")
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        from vllm.attention.ops.triton_paged_attn import triton_paged_attention
        output = triton_paged_attention(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            kv_cache, attn_metadata["slot_mapping"], attn_metadata["seq_lens"], None, self.scale
        )
        # Reshape based on actual output elements to support 35B's varying hidden sizes
        attn_out_flat = output.reshape(-1, output.shape[-1] * output.shape[-2])
        return self.o_proj(attn_out_flat).view(hidden_states.shape[0], hidden_states.shape[1], -1)

class Qwen3_5MLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down")
        self.act = Silu()
    def forward(self, x):
        gate_up = self.gate_up_proj(x); gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)

class Qwen3_5Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3_5Attention(config, layer_id, quant_config, prefix)
        if hasattr(config, "num_experts") and config.num_experts > 0: self.mlp = Qwen3_5MoE(config, quant_config, prefix)
        else: self.mlp = Qwen3_5MLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), positions, kv_cache, attn_metadata)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

class Qwen3_5Model(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.norm(x)

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen3_5Model(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        return self.lm_head(self.model(input_ids, positions, kv_caches, attn_metadata))

class Qwen3_5MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.num_experts = config.num_experts; self.topk = config.num_experts_per_tok
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_exps")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_exps")
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False).cuda().half()
    def forward(self, x):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        self.w1(x[:1]); self.w2(torch.zeros((1, self.w2.input_size), device=x.device, dtype=x.dtype))
        w1_t = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id); w2_t = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)
        return fused_moe(x.view(-1, x.shape[-1]), w1_t.view(self.num_experts, -1, x.shape[-1]), w2_t.view(self.num_experts, x.shape[-1], -1), self.gate(x.view(-1, x.shape[-1])), topk=self.topk).view(x.shape)

class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM): pass
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForCausalLM): pass
class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM): pass
