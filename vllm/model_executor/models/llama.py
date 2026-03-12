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

def get_rotary_embedding(config: LiteConfig):
    from vllm.model_executor.layers.rotary_embedding.base import get_rope
    head_size = config.hidden_size // config.num_attention_heads
    return get_rope(head_size=head_size, rotary_dim=head_size, max_position=config.max_position_embeddings, 
                    base=config.rope_theta, is_neox_style=True)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = type('obj', (object,), {
            'num_heads': config.num_attention_heads, 'num_kv_heads': config.num_key_value_heads,
            'head_dim': config.hidden_size // config.num_attention_heads,
            'scale': (config.hidden_size // config.num_attention_heads)**-0.5,
            'q_size': config.num_attention_heads * (config.hidden_size // config.num_attention_heads),
            'kv_size': config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        })
        # Use STANDARD naming for GGUF/HF compatibility (model.layers.i.self_attn.q_proj)
        self.self_attn.q_proj = LiteLinear(config.hidden_size, self.self_attn.q_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj")
        self.self_attn.k_proj = LiteLinear(config.hidden_size, self.self_attn.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.k_proj")
        self.self_attn.v_proj = LiteLinear(config.hidden_size, self.self_attn.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.v_proj")
        self.self_attn.o_proj = LiteLinear(self.self_attn.q_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.rotary_emb = get_rotary_embedding(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.mlp = type('obj', (object,), {
            'gate_proj': LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj"),
            'up_proj': LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj"),
            'down_proj': LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
        })

    def forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None):
        if hasattr(self, "_fast_forward") and self._fast_forward is not None:
            return self._fast_forward(x, positions, kv_cache, attn_metadata)
        return self._standard_forward(x, positions, kv_cache, attn_metadata, lora_mapping)

    def _standard_forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(x)
        q = self.self_attn.q_proj(h); k = self.self_attn.k_proj(h); v = self.self_attn.v_proj(h)
        bs, seq = q.shape[:2]
        q = q.view(bs, seq, self.self_attn.num_heads, self.self_attn.head_dim); k = k.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim); v = v.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        slot_mapping = getattr(attn_metadata, "slot_mapping", None) if not isinstance(attn_metadata, dict) else attn_metadata.get("slot_mapping")
        if slot_mapping is not None and kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            k_cache, v_cache = kv_cache
            reshape_and_cache(k.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), v.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), k_cache, v_cache, slot_mapping, "auto")
            attn_in = torch.empty((bs * seq, self.self_attn.num_heads, self.self_attn.head_dim), device=q.device, dtype=q.dtype)
            paged_attention_v1(attn_in, q.reshape(bs * seq, self.self_attn.num_heads, self.self_attn.head_dim).contiguous(), k_cache, v_cache, self.self_attn.num_kv_heads, self.self_attn.scale, getattr(attn_metadata, "block_tables", None), getattr(attn_metadata, "seq_lens", None), k_cache.shape[1], 4096, None, "auto", None, None)
            out = attn_in.view(bs, seq, -1)
        else: out = q.view(bs, seq, -1)
        hidden_states = x + self.self_attn.o_proj(out)
        h = self.post_attention_layernorm(hidden_states); gate = self.mlp.gate_proj(h); up = self.mlp.up_proj(h)
        return hidden_states + self.mlp.down_proj(F.silu(gate) * up)

    def compile_fast_dispatch(self):
        self._fast_forward = None

class LlamaModel(nn.Module):
    def __init__(self, hf_config, quant_config, prefix="model"):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # Fix: ensure prefix join is clean (no model..layers)
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, quant_config, f"{prefix}.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        if input_ids.dtype == torch.long: x = self.embed_tokens(input_ids)
        else: x = input_ids
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix="model"):
        super().__init__()
        self.model = LlamaModel(vllm_config.model_config.hf_config, vllm_config.quant_config, "model")
        self.lm_head = LiteLinear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states[:, -1:, :])
