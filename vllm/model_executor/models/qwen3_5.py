# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
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
        h = self.input_layernorm(x)
        q = self.q_proj(h); k = self.k_proj(h); v = self.v_proj(h)
        bs, seq = q.shape[:2]
        q, k = self.rotary_emb(positions, q.view(bs,seq,self.num_heads,-1), k.view(bs,seq,self.num_kv_heads,-1))
        return self.o_proj(q.reshape(bs,seq,-1)) + x

    def compile_fast_dispatch(self):
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.model_executor.layers.quantization.gguf_kernels import ggml_mul_mat_a8_fallback
        
        TILE_AWQ = 1024; TILE_GGUF = 1152 # 36 blocks of 32

        def _get_tiles(proj):
            q_type, d = proj.get_fast_data()
            if q_type == "fp16": return "fp16", d[0]
            
            tiles = []
            if q_type == "awq":
                qw, sc, qz, gs = d
                for s in range(0, proj.input_size, TILE_AWQ):
                    e = min(s + TILE_AWQ, proj.input_size)
                    tiles.append((qw[:, s//8:e//8].contiguous(), sc[:, s//gs:e//gs].contiguous(), qz[:, s//gs//8:e//gs//8+1].contiguous() if qz is not None else None, gs, s, e))
                return "awq", tiles
            
            if q_type == "gguf":
                qw, sc, qt, pref = d
                for s in range(0, proj.input_size, TILE_GGUF):
                    e = min(s + TILE_GGUF, proj.input_size)
                    # GGUF Q4_0: 18 bytes per 32 weights
                    qw_t = qw[:, (s//32)*18 : (e//32)*18].contiguous()
                    tiles.append((qw_t, sc, qt, proj.output_size, s, e))
                return "gguf", tiles
            return "fp16", d[0]

        q_t_type, q_t = _get_tiles(self.q_proj); k_t_type, k_t = _get_tiles(self.k_proj); v_t_type, v_t = _get_tiles(self.v_proj)
        g_t_type, g_t = _get_tiles(self.gate_proj); u_t_type, u_t = _get_tiles(self.up_proj)
        o_type, o_d = self.o_proj.get_fast_data(); d_type, d_d = self.down_proj.get_fast_data()
        in_w = self.input_layernorm.weight; post_w = self.post_attention_layernorm.weight; eps = self.input_layernorm.variance_epsilon
        rotary = self.rotary_emb; n_h = self.num_heads; n_kv = self.num_kv_heads; h_d = self.head_dim; scale = self.scale

        def _run_op(x, t_type, tiles):
            if t_type == "fp16": return F.linear(x, tiles)
            if t_type == "awq": return sum(awq_fused_gemm(x[:, t[4]:t[5]].contiguous(), *t[:4]) for t in tiles)
            if t_type == "gguf": return sum(ggml_mul_mat_a8_fallback(t[0], x[:, t[4]:t[5]].contiguous(), t[2], t[3]) for t in tiles)
            return 0

        @torch.inference_mode()
        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            bs = hidden_states.shape[0]; h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            h_norm = F.rms_norm(h_flat, (h_flat.shape[-1],), in_w, eps)
            
            q = _run_op(h_norm, q_t_type, q_t); k = _run_op(h_norm, k_t_type, k_t); v = _run_op(h_norm, v_t_type, v_t)
            q = q.view(bs, 1, n_h, h_d).contiguous(); k = k.view(bs, 1, n_kv, h_d).contiguous(); v = v.view(bs, 1, n_kv, h_d).contiguous()
            q, k = rotary(positions, q, k)
            
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            reshape_and_cache(k.view(-1, n_kv, h_d).contiguous(), v.view(-1, n_kv, h_d).contiguous(), kv_cache[0], kv_cache[1], attn_metadata.slot_mapping, "auto")
            attn_in = torch.empty((bs, n_h, h_d), device=q.device, dtype=q.dtype)
            paged_attention_v1(attn_in, q.view(bs, n_h, h_d).contiguous(), kv_cache[0], kv_cache[1], n_kv, scale, attn_metadata.block_tables, attn_metadata.seq_lens, kv_cache[0].shape[1], 4096, None, "auto")
            
            # Output & MLP
            attn_out = _run_op(attn_in.view(bs, -1).contiguous(), o_type, o_d if o_type=="fp16" else [(*o_d, 0, attn_in.view(bs,-1).shape[-1])])
            hidden_states = hidden_states + attn_out.view(hidden_states.shape)
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            h_norm_post = F.rms_norm(h_flat, (h_flat.shape[-1],), post_w, eps)
            gate = _run_op(h_norm_post, g_t_type, g_t); up = _run_op(h_norm_post, u_t_type, u_t)
            mlp_out = _run_op((F.silu(gate) * up).contiguous(), d_type, d_d if d_type=="fp16" else [(*d_d, 0, gate.shape[-1])])
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
