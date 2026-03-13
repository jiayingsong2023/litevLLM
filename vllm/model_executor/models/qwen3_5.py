# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.utils import LiteBufferManager
from .lite_config import LiteConfig

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config; self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.q_proj = LiteLinear(config.hidden_size, config.num_attention_heads * (config.hidden_size // config.num_attention_heads), bias=True, quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.k_proj = LiteLinear(config.hidden_size, config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), bias=True, quant_config=quant_config, prefix=f"{prefix}.k_proj")
        self.v_proj = LiteLinear(config.hidden_size, config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), bias=True, quant_config=quant_config, prefix=f"{prefix}.v_proj")
        self.o_proj = LiteLinear(config.num_attention_heads * (config.hidden_size // config.num_attention_heads), config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.num_heads = config.num_attention_heads; self.num_kv_heads = config.num_key_value_heads; self.head_dim = config.hidden_size // self.num_heads; self.scale = self.head_dim**-0.5
        from .llama import get_rotary_embedding
        self.rotary_emb = get_rotary_embedding(config)

    def forward(self, x, positions, kv_cache, attn_metadata):
        if hasattr(self, "_fast_forward"): return self._fast_forward(x, positions, kv_cache, attn_metadata)
        h = self.input_layernorm(x); q = self.q_proj(h); return self.o_proj(q) + x

    def compile_fast_dispatch(self):
        """Ultra-Performance Closure with Pre-translated PagedAttention Caching."""
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.kernels.triton.gguf_tiled_fused_gemm import gguf_fused_gemm
        from vllm.kernels.triton.lite_rmsnorm import lite_rmsnorm
        
        buf_mgr = LiteBufferManager()
        buf_mgr.init_pool(max_batch=32, max_hidden=self.config.hidden_size, max_intermediate=32768, device="cuda")

        def _get_op(proj):
            q_type, d = proj.get_fast_data()
            if q_type == "awq": return lambda x, out=None: awq_fused_gemm(x, *d, out=out)
            if q_type == "gguf": return lambda x, out=None: gguf_tiled_fused_gemm.gguf_fused_gemm(x, d[0], out=out)
            return lambda x, out=None: F.linear(x, d[0], out=out)

        self._q_op = _get_op(self.q_proj); self._k_op = _get_op(self.k_proj); self._v_op = _get_op(self.v_proj)
        self._o_op = _get_op(self.o_proj); self._gate_op = _get_op(self.gate_proj); self._up_op = _get_op(self.up_proj); self._down_op = _get_op(self.down_proj)
        in_w = self.input_layernorm.weight; post_w = self.post_attention_layernorm.weight; eps = self.input_layernorm.variance_epsilon
        rotary = self.rotary_emb; n_h = self.num_heads; n_kv = self.num_kv_heads; h_d = self.head_dim; scale = self.scale; h_size = self.config.hidden_size; i_size = self.config.intermediate_size

        @torch.inference_mode()
        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            bs = hidden_states.shape[0]
            
            # --- 1. PRE-TRANSLATE PHYSICAL POINTERS (Once per Layer or Step) ---
            # Extract base pointers for KV Cache
            k_cache, v_cache = kv_cache
            k_base = k_cache.data_ptr(); v_base = v_cache.data_ptr()
            
            # Stride info
            k_block_stride = k_cache.stride(0) * k_cache.element_size()
            v_block_stride = v_cache.stride(0) * v_cache.element_size()
            
            # Calculate physical pointers for all blocks in the batch
            # Note: For production, we'd use a small Triton kernel to do this faster
            bt = attn_metadata.block_tables
            max_blocks = bt.shape[1]
            k_ptrs = k_base + bt.to(torch.int64) * k_block_stride
            v_ptrs = v_base + bt.to(torch.int64) * v_block_stride
            
            # --- 2. ATTENTION STAGE ---
            h_norm = buf_mgr.get_slice("a", bs, h_size, offset_dim=0)
            lite_rmsnorm(hidden_states.view(-1, h_size), in_w, eps, out=h_norm)
            
            q_buf = buf_mgr.get_slice("b", bs, n_h * h_d, offset_dim=0)
            k_buf = buf_mgr.get_slice("b", bs, n_kv * h_d, offset_dim=h_size)
            v_buf = buf_mgr.get_slice("b", bs, n_kv * h_d, offset_dim=h_size * 2)
            
            self._q_op(h_norm, out=q_buf); self._k_op(h_norm, out=k_buf); self._v_op(h_norm, out=v_buf)
            
            q = q_buf.view(bs, 1, n_h, h_d).contiguous(); k = k_buf.view(bs, 1, n_kv, h_d).contiguous(); v = v_buf.view(bs, 1, n_kv, h_d).contiguous()
            q, k = rotary(positions, q, k)
            
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            reshape_and_cache(k.view(-1, n_kv, h_d).contiguous(), v.view(-1, n_kv, h_d).contiguous(), k_cache, v_cache, attn_metadata.slot_mapping, "auto")
            
            attn_out = buf_mgr.get_slice_3d("a", bs, n_h, h_d, offset_dim=0)
            # CALLING PAGED ATTENTION V1 WITH CACHED PTRS
            paged_attention_v1(attn_out, q.view(bs, n_h, h_d).contiguous(), k_cache, v_cache, n_kv, scale, attn_metadata.block_tables, attn_metadata.seq_lens, k_cache.shape[1], 4096, None, "auto", k_ptrs=k_ptrs, v_ptrs=v_ptrs)
            
            hidden_states = hidden_states + self._o_op(attn_out.view(bs, -1).contiguous()).view(hidden_states.shape)
            
            # --- 3. MLP STAGE ---
            h_norm_post = buf_mgr.get_slice("a", bs, h_size, offset_dim=0)
            lite_rmsnorm(hidden_states.view(-1, h_size), post_w, eps, out=h_norm_post)
            gate_buf = buf_mgr.get_slice("b", bs, i_size, offset_dim=0); up_buf = buf_mgr.get_slice("b", bs, i_size, offset_dim=i_size)
            self._gate_op(h_norm_post, out=gate_buf); self._up_op(h_norm_post, out=up_buf)
            
            mlp_out = self._down_op((F.silu(gate_buf) * up_buf).contiguous())
            return hidden_states + mlp_out.view(hidden_states.shape)
            
        self._fast_forward = _fast_forward

class Qwen2Model(nn.Module):
    def __init__(self, hf_config, quant_config):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(self.config, quant_config, f"model.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))

class Qwen3_5ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen2Model(vllm_config.model_config.hf_config, vllm_config.quant_config)
        self.lm_head = LiteLinear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, prefix="lm_head")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.model.embed_tokens(input_ids)
        for i in range(len(self.model.layers)): x = self.model.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.lm_head(self.model.norm(x)[:, -1:, :])

class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration): pass
