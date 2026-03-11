# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu

class LlamaAttention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.hidden_size = config.hidden_size; self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads; self.head_dim = self.hidden_size // self.num_heads
        self.qkv_proj = LiteLinear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.qkv_proj")
        self.o_proj = LiteLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, lora_mapping=lora_mapping)
        q, k, v = qkv.split([self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim); v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        k_cache_storage, v_cache_storage = kv_cache 
        slot_mapping = attn_metadata["slot_mapping"]; kv_start_indices = attn_metadata["kv_start_indices"]; seq_lens = attn_metadata["seq_lens"]
        
        target_heads, target_dim = k_cache_storage.shape[2], k_cache_storage.shape[3]
        for i in range(bsz):
            slot = slot_mapping[i]; start = kv_start_indices[i]; end = start + seqlen
            k_cache_storage[slot, start:end, :, :] = k[i, :, :target_heads, :target_dim].to(k_cache_storage.dtype)
            v_cache_storage[slot, start:end, :, :] = v[i, :, :target_heads, :target_dim].to(v_cache_storage.dtype)
        if seqlen > 1: torch.cuda.synchronize()

        if seqlen > 1:
            q = q.transpose(1, 2); max_l = seq_lens.max().item()
            k_hist = k_cache_storage[slot_mapping, :max_l].to(q.dtype).transpose(1, 2)
            v_hist = v_cache_storage[slot_mapping, :max_l].to(q.dtype).transpose(1, 2)
            if target_heads != self.num_heads:
                n_rep = self.num_heads // target_heads
                k_hist, v_hist = k_hist.repeat_interleave(n_rep, dim=1), v_hist.repeat_interleave(n_rep, dim=1)
            output = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=True, scale=self.scale)
            output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        else:
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            max_len = k_cache_storage.shape[1]; block_tables = slot_mapping.view(bsz, 1).to(torch.int32)
            output = torch.empty((bsz, self.num_heads, self.head_dim), device=q.device, dtype=q.dtype)
            paged_attention_v1(output, q.view(bsz, self.num_heads, self.head_dim), k_cache_storage, v_cache_storage, target_heads, self.scale, block_tables, seq_lens, block_size=max_len, max_seq_len=max_len, alibi_slopes=None, kv_cache_dtype="fp8" if k_cache_storage.dtype == torch.float8_e4m3fn else "fp16", k_scale=None, v_scale=None)
            output = output.view(bsz, 1, -1)
        return self.o_proj(output, lora_mapping=lora_mapping)

class LlamaModel(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i, vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        if self.embed_tokens.weight.device != input_ids.device: x = self.embed_tokens(input_ids.cpu()).to(input_ids.device)
        else: x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(x)

class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        return self.down_proj(self.act(self.gate_up_proj(x, lora_mapping=lora_mapping)), lora_mapping=lora_mapping)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_id, quant_config, prefix)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        # 1. Check for Fast Dispatch path (pre-bound)
        if hasattr(self, "_fast_forward"):
            return self._fast_forward(hidden_states, positions, kv_cache, attn_metadata)

        h = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping)
        return hidden_states

    def compile_fast_dispatch(self):
        """
        Pre-binds weight pointers and Triton kernels to minimize Python overhead.
        """
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.kernels.triton.rmsnorm_awq_fused import rmsnorm_awq_fused_linear
        from vllm.model_executor.layers.quantization.tensor import AWQWeight
        
        # Pre-extract weights for AWQ
        def _ensure_and_get_awq_data(proj):
            if hasattr(proj, "quant_config") and proj.quant_config:
                 device = proj.qweight.device if hasattr(proj, "qweight") and proj.qweight is not None else "cuda"
                 dummy_x = torch.zeros((1, proj.input_size), device=device, dtype=torch.float16)
                 proj.quant_config.apply(proj, dummy_x)
            
            if isinstance(proj.weight, AWQWeight):
                return proj.weight.qweight, proj.weight.scales, proj.weight.qzeros, proj.weight.group_size
            return None

        # 1. Attention Pre-binds (Fused RMSNorm + QKV)
        self._input_norm_w = self.input_layernorm.weight
        self._input_norm_eps = self.input_layernorm.variance_epsilon
        self._qkv_data = _ensure_and_get_awq_data(self.self_attn.qkv_proj)
        self._o_data = _ensure_and_get_awq_data(self.self_attn.o_proj)
        
        # 2. MLP Pre-binds (Fused Post-Norm + GateUp)
        self._post_norm_w = self.post_attention_layernorm.weight
        self._post_norm_eps = self.post_attention_layernorm.variance_epsilon
        self._gate_up_data = _ensure_and_get_awq_data(self.mlp.gate_up_proj)
        self._down_data = _ensure_and_get_awq_data(self.mlp.down_proj)

        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            # A. Attention Block (Macro-Fusion 1)
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            # RMSNorm + QKV Proj Fused
            qkv = rmsnorm_awq_fused_linear(h_flat, *self._qkv_data, self._input_norm_w, self._input_norm_eps)
            
            # O-proj (Standard Fused AWQ for now)
            attn_out = awq_fused_gemm(qkv[..., :hidden_states.shape[-1]], *self._o_data)
            hidden_states = hidden_states + attn_out.view(hidden_states.shape)
            
            # B. MLP Block (Macro-Fusion 2)
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            # RMSNorm + GateUp Proj Fused
            gate_up = rmsnorm_awq_fused_linear(h_flat, *self._gate_up_data, self._post_norm_w, self._post_norm_eps)
            
            gate, up = gate_up.chunk(2, dim=-1)
            mlp_out = awq_fused_gemm(torch.nn.functional.silu(gate) * up, *self._down_data)
            hidden_states = hidden_states + mlp_out.view(hidden_states.shape)
            
            return hidden_states

        self._fast_forward = _fast_forward

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = LlamaModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        # ONLY PROJECT THE LAST TOKEN
        # hidden_states: (Batch, Seq, Hidden) -> (Batch, 1, Hidden)
        last_hidden = hidden_states[:, -1:, :]
        return self.lm_head(last_hidden, lora_mapping=lora_mapping)
