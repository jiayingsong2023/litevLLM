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

class GLMAttention(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        # Official GLM GGUF Naming
        self.query_key_value = LiteLinear(config.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.self_attn.query_key_value")
        self.dense = LiteLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.dense")

    def forward(self, x, kv_cache, attn_metadata):
        qkv = self.query_key_value(x)
        bs, seq, _ = x.shape
        n_tokens = bs * seq
        # GLM architecture: qkv splitting
        q = qkv[..., :self.num_heads * self.head_dim]
        k = qkv[..., self.num_heads * self.head_dim : (self.num_heads + self.num_kv_heads) * self.head_dim]
        v = qkv[..., (self.num_heads + self.num_kv_heads) * self.head_dim :]
        
        k_cache, v_cache = kv_cache
        from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
        from vllm.kernels.triton.paged_attention import paged_attention_v1
        
        # Mapping to KV cache
        slot_mapping = attn_metadata["slot_mapping"]
        reshape_and_cache(k.view(-1, self.num_kv_heads, self.head_dim), 
                          v.view(-1, self.num_kv_heads, self.head_dim), 
                          k_cache, v_cache, slot_mapping, "auto")
        
        attn_out = torch.empty((n_tokens, self.num_heads, self.head_dim), device=q.device, dtype=q.dtype)
        block_tables = attn_metadata["block_tables"]
        seq_lens = attn_metadata["seq_lens"]

        if seq > 1:
            end_pos = seq_lens[0].item()
            start_pos = end_pos - seq
            seq_lens_ext = torch.arange(start_pos + 1, end_pos + 1, device=q.device, dtype=torch.int32)
            block_tables_ext = block_tables.expand(seq, -1).contiguous()
            paged_attention_v1(attn_out, q.view(n_tokens, self.num_heads, self.head_dim).contiguous(), 
                              k_cache, v_cache, self.num_kv_heads, self.head_dim**-0.5, 
                              block_tables_ext, seq_lens_ext, 16, 4096, None, "auto")
        else:
            paged_attention_v1(attn_out, q.view(n_tokens, self.num_heads, self.head_dim).contiguous(), 
                              k_cache, v_cache, self.num_kv_heads, self.head_dim**-0.5, 
                              block_tables, seq_lens, 16, 4096, None, "auto")
        
        return self.dense(attn_out.view(bs, seq, -1))

class GLMDecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config; eps = _get_eps(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = GLMAttention(config, quant_config, prefix=prefix)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = type('obj', (object,), {
            'gate_up_proj': LiteLinear(config.hidden_size, config.intermediate_size * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.dense_h_to_4h"),
            'down_proj': LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.dense_4h_to_h")
        })

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        if hasattr(self, "_fast_forward") and self._fast_forward is not None:
            return self._fast_forward(hidden_states, positions, kv_cache, attn_metadata)
        h = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(h, kv_cache, attn_metadata)
        hidden_states = hidden_states + attn_out; h = self.post_attention_layernorm(hidden_states)
        gate_up = self.mlp.gate_up_proj(h); gate, up = gate_up.chunk(2, dim=-1)
        return hidden_states + self.mlp.down_proj(F.silu(gate) * up)

    def compile_fast_dispatch(self):
        self._fast_forward = None

class GLMModel(nn.Module):
    def __init__(self, hf_config, quant_config, prefix="model"):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([GLMDecoderLayer(self.config, quant_config, f"{prefix}.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.norm(x)

class GlmForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix="model"):
        super().__init__()
        p = prefix if prefix.endswith(".") or prefix == "" else f"{prefix}."
        self.model = GLMModel(vllm_config.model_config.hf_config, vllm_config.quant_config, "model")
        self.lm_head = LiteLinear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states[:, -1:, :])

class Glm4MoeLiteForCausalLM(GlmForCausalLM): pass
