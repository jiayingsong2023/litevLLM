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
    def __init__(self, config: LiteConfig, quant_config, prefix="", layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = nn.Module()
        self.self_attn.num_heads = config.num_attention_heads
        self.self_attn.num_kv_heads = config.num_key_value_heads
        self.self_attn.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn.scale = (self.self_attn.head_dim)**-0.5
        self.self_attn.q_size = self.self_attn.num_heads * self.self_attn.head_dim
        self.self_attn.kv_size = self.self_attn.num_kv_heads * self.self_attn.head_dim
        
        # Use STANDARD naming for GGUF/HF compatibility (model.layers.i.self_attn.q_proj)
        self.self_attn.q_proj = LiteLinear(config.hidden_size, self.self_attn.q_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj")
        self.self_attn.k_proj = LiteLinear(config.hidden_size, self.self_attn.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.k_proj")
        self.self_attn.v_proj = LiteLinear(config.hidden_size, self.self_attn.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.v_proj")
        self.self_attn.o_proj = LiteLinear(self.self_attn.q_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        
        self.rotary_emb = get_rotary_embedding(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        
        self.mlp = nn.Module()
        self.mlp.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj")
        self.mlp.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj")
        self.mlp.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")

    def forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None):
        if hasattr(self, "_fast_forward") and self._fast_forward is not None:
            return self._fast_forward(x, positions, kv_cache, attn_metadata)
        return self._standard_forward(x, positions, kv_cache, attn_metadata, lora_mapping)

    def _standard_forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(x)
        q = self.self_attn.q_proj(h, lora_mapping); k = self.self_attn.k_proj(h, lora_mapping); v = self.self_attn.v_proj(h, lora_mapping)
        bs, seq = q.shape[:2]
        q = q.view(bs, seq, self.self_attn.num_heads, self.self_attn.head_dim); k = k.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim); v = v.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        slot_mapping = getattr(attn_metadata, "slot_mapping", None) if not isinstance(attn_metadata, dict) else attn_metadata.get("slot_mapping")
        if slot_mapping is not None and kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            
            inf_config = attn_metadata.get("config") if isinstance(attn_metadata, dict) else getattr(attn_metadata, "config", None)
            kv_cache_dtype = inf_config.kv_type if inf_config else (attn_metadata.get("kv_cache_dtype", "auto") if isinstance(attn_metadata, dict) else getattr(attn_metadata, "kv_cache_dtype", "auto"))
            k_scale = inf_config.k_scale if inf_config else (attn_metadata.get("k_scale", 1.0) if isinstance(attn_metadata, dict) else getattr(attn_metadata, "k_scale", 1.0))
            v_scale = inf_config.v_scale if inf_config else (attn_metadata.get("v_scale", 1.0) if isinstance(attn_metadata, dict) else getattr(attn_metadata, "v_scale", 1.0))

            k_cache, v_cache = kv_cache
            kv_scale_cache = attn_metadata.get("kv_scale_cache")
            if kv_scale_cache is not None:
                k_scale_cache, v_scale_cache = kv_scale_cache[self.layer_idx]
            else:
                k_scale_cache, v_scale_cache = (None, None)

            reshape_and_cache(k.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), 
                              v.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), 
                              k_cache, v_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale,
                              k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache)
            
            attn_in = torch.empty((bs * seq, self.self_attn.num_heads, self.self_attn.head_dim), device=q.device, dtype=q.dtype)
            block_tables = attn_metadata.get("block_tables", None) if isinstance(attn_metadata, dict) else getattr(attn_metadata, "block_tables", None)
            seq_lens = attn_metadata.get("seq_lens", None) if isinstance(attn_metadata, dict) else getattr(attn_metadata, "seq_lens", None)
            is_prefill = attn_metadata.get("is_prefill", False) if isinstance(attn_metadata, dict) else getattr(attn_metadata, "is_prefill", False)
            
            from vllm.engine.lite_engine import expand_metadata_for_paged_attention
            
            # CRITICAL: For chunked prefill (seq > 1), we must expand metadata for the kernel.
            max_ctx = int(
                max(
                    self.self_attn.num_heads * self.self_attn.head_dim,
                    getattr(self.config, "max_position_embeddings", 4096),
                )
            )
            
            seq_lens_ext, block_tables_ext = expand_metadata_for_paged_attention(
                bs, seq, is_prefill, seq_lens, block_tables, q.device
            )
            
            # For paged_attention, we need per-block scale pointers if using row-scales
            # In LiteEngine, we can pass the scale caches directly as they match KV layout.
            paged_attention_v1(
                attn_in,
                q.reshape(bs * seq, self.self_attn.num_heads, self.self_attn.head_dim).contiguous(),
                k_cache,
                v_cache,
                self.self_attn.num_heads,
                self.self_attn.scale,
                block_tables_ext,
                seq_lens_ext,
                k_cache.shape[1],
                max_ctx,
                None,
                kv_cache_dtype,
                k_scale,
                v_scale,
                k_scale_ptrs=k_scale_cache, # Passing full tensor; kernel uses block_idx/stride
                v_scale_ptrs=v_scale_cache,
                num_kv_heads=self.self_attn.num_kv_heads,
            )
            out = attn_in.view(bs, seq, -1)
        else: out = q.view(bs, seq, -1)
        hidden_states = x + self.self_attn.o_proj(out, lora_mapping)
        h = self.post_attention_layernorm(hidden_states)
        gate = self.mlp.gate_proj(h, lora_mapping)
        up = self.mlp.up_proj(h, lora_mapping)
        mlp_out = self.mlp.down_proj(F.silu(gate) * up, lora_mapping)
        return hidden_states + mlp_out

    def compile_fast_dispatch(self):
        self._fast_forward = None

class LlamaModel(nn.Module):
    def __init__(self, hf_config, quant_config, prefix="model"):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # Standard: model.embed_tokens, model.layers.0.xxx, model.norm
        # If prefix is 'model', name of layer 0's q_proj will be 'layers.0.q_proj' within this module
        # so parameters are model.embed_tokens, model.layers.0.xxx
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, quant_config, f"layers.{i}", layer_idx=i) for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        if input_ids.dtype == torch.long: x = self.embed_tokens(input_ids)
        else: x = input_ids
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        # HF standard: the backbone is often called 'model'
        # so parameters are model.embed_tokens, model.layers.0.xxx
        self.model = LlamaModel(vllm_config.model_config.hf_config, vllm_config.quant_config, "model")
        self.lm_head = LiteLinear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping)
        return self.lm_head(hidden_states[:, -1:, :], lora_mapping)
