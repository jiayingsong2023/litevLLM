# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.config import VllmConfig
from .lite_config import LiteConfig

class Qwen2Attention(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        self.q_proj = LiteLinear(config.hidden_size, self.q_size, bias=True, quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.k_proj = LiteLinear(config.hidden_size, self.kv_size, bias=True, quant_config=quant_config, prefix=f"{prefix}.k_proj")
        self.v_proj = LiteLinear(config.hidden_size, self.kv_size, bias=True, quant_config=quant_config, prefix=f"{prefix}.v_proj")
        self.o_proj = LiteLinear(self.q_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")

class Qwen2MLP(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config; self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.self_attn = Qwen2Attention(config, quant_config, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.mlp = Qwen2MLP(config, quant_config, prefix=f"{prefix}.mlp")
        from .llama import get_rotary_embedding
        self.rotary_emb = get_rotary_embedding(config)

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        if hasattr(self, "_fast_forward"): return self._fast_forward(hidden_states, positions, kv_cache, attn_metadata)
        return self._standard_forward(hidden_states, positions, kv_cache, attn_metadata)

    def _standard_forward(self, hidden_states, positions, kv_cache, attn_metadata):
        h = self.input_layernorm(hidden_states)
        q = self.self_attn.q_proj(h); k = self.self_attn.k_proj(h); v = self.self_attn.v_proj(h)
        bs, seq = q.shape[:2]
        q = q.view(bs, seq, self.self_attn.num_heads, self.self_attn.head_dim); k = k.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim); v = v.view(bs, seq, self.self_attn.num_kv_heads, self.self_attn.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        if kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            reshape_and_cache(k.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), v.reshape(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous(), kv_cache[0], kv_cache[1], attn_metadata.slot_mapping, "auto")
            attn_in = torch.empty((bs * seq, self.self_attn.num_heads, self.self_attn.head_dim), device=q.device, dtype=q.dtype)
            paged_attention_v1(attn_in, q.reshape(bs * seq, self.self_attn.num_heads, self.self_attn.head_dim).contiguous(), kv_cache[0], kv_cache[1], self.self_attn.num_kv_heads, self.self_attn.scale, attn_metadata.block_tables, attn_metadata.seq_lens, kv_cache[0].shape[1], 4096, None, "auto", None, None)
            out = attn_in.view(bs, seq, -1)
        else: out = q.view(bs, seq, -1)
        hidden_states = hidden_states + self.self_attn.o_proj(out); h = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))

    def compile_fast_dispatch(self):
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.kernels.triton.rmsnorm_awq_fused import rmsnorm_awq_fused_linear
        from vllm.model_executor.layers.quantization.tensor import AWQWeight
        def _get_data(proj):
            w = getattr(proj, "_quant_weight", None)
            if isinstance(w, AWQWeight): return w.qweight, w.scales, w.qzeros, w.group_size
            return None
        self._q_data = _get_data(self.self_attn.q_proj); self._k_data = _get_data(self.self_attn.k_proj); self._v_data = _get_data(self.self_attn.v_proj)
        self._o_data = _get_data(self.self_attn.o_proj); self._gate_data = _get_data(self.mlp.gate_proj); self._up_data = _get_data(self.mlp.up_proj); self._down_data = _get_data(self.mlp.down_proj)
        self._input_norm_w = self.input_layernorm.weight; self._post_norm_w = self.post_attention_layernorm.weight; self._eps = self.input_layernorm.variance_epsilon
        if self._q_data is None: return

        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            bs = hidden_states.shape[0]; h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            q = rmsnorm_awq_fused_linear(h_flat, *self._q_data, self._input_norm_w, self._eps)
            k = awq_fused_gemm(h_flat, *self._k_data); v = awq_fused_gemm(h_flat, *self._v_data)
            q = q.view(bs, 1, self.self_attn.num_heads, self.self_attn.head_dim).contiguous()
            k = k.view(bs, 1, self.self_attn.num_kv_heads, self.self_attn.head_dim).contiguous()
            q, k = self.rotary_emb(positions, q, k)
            # PagedAttention
            attn_in = torch.empty((bs, self.self_attn.num_heads, self.self_attn.head_dim), device=q.device, dtype=q.dtype)
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            paged_attention_v1(attn_in, q.view(bs, self.self_attn.num_heads, self.self_attn.head_dim).contiguous(), kv_cache[0], kv_cache[1], self.self_attn.num_kv_heads, self.self_attn.scale, attn_metadata.block_tables, attn_metadata.seq_lens, kv_cache[0].shape[1], 4096, None, "auto", None, None)
            attn_out = awq_fused_gemm(attn_in.view(bs, -1).contiguous(), *self._o_data)
            hidden_states = hidden_states + attn_out.view(hidden_states.shape); h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            gate = rmsnorm_awq_fused_linear(h_flat, *self._gate_data, self._post_norm_w, self._eps)
            up = awq_fused_gemm(h_flat, *self._up_data)
            mlp_out = awq_fused_gemm((F.silu(gate) * up).contiguous(), *self._down_data)
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
