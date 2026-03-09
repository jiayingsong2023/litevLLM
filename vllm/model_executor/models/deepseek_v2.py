# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu

class DeepSeekV2Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.config = config; self.hidden_size = config.hidden_size; self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", getattr(config, "head_dim", 128))
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "v_head_dim", getattr(config, "head_dim", 128))
        self.q_proj = LiteLinear(self.hidden_size, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_q")
        self.kv_proj = LiteLinear(self.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_kv_a_mqa")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_output")
        self.scale = self.qk_nope_head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states, lora_mapping=lora_mapping).view(bsz, seq_len, self.num_heads, -1)
        kv = self.kv_proj(hidden_states, lora_mapping=lora_mapping)
        actual_q_nope_dim = q.shape[-1] - self.qk_rope_head_dim
        q_nope = q[..., :actual_q_nope_dim]
        k = kv[..., :actual_q_nope_dim].view(bsz, seq_len, 1, actual_q_nope_dim)
        v = kv[..., actual_q_nope_dim:].view(bsz, seq_len, 1, -1)
        actual_v_dim = v.shape[-1]

        k_cache_storage, v_cache_storage = kv_cache 
        slot_mapping = attn_metadata["slot_mapping"]
        kv_start_indices = attn_metadata["kv_start_indices"]; seq_lens = attn_metadata["seq_lens"]
        
        target_heads_k, target_dim_k = k_cache_storage.shape[2], k_cache_storage.shape[3]
        target_heads_v, target_dim_v = v_cache_storage.shape[2], v_cache_storage.shape[3]

        def align_t(t, th, td):
            b, l, h, d = t.shape
            if h != th or d != td:
                t = t.reshape(b, l, -1)
                req = th * td
                if t.shape[-1] > req: t = t[..., :req]
                else: t = F.pad(t, (0, req - t.shape[-1]))
                return t.view(b, l, th, td)
            return t
        k = align_t(k, target_heads_k, target_dim_k); v = align_t(v, target_heads_v, target_dim_v)

        for i in range(bsz):
            slot = slot_mapping[i]; start = kv_start_indices[i]; end = start + seq_len
            k_cache_storage[slot, start:end, :, :] = k[i].to(k_cache_storage.dtype)
            v_cache_storage[slot, start:end, :, :] = v[i].to(v_cache_storage.dtype)
        if seq_len > 1: torch.cuda.synchronize()

        if seq_len > 1:
            max_l = seq_lens.max().item()
            k_hist = k_cache_storage[slot_mapping, :max_l].to(q_nope.dtype)
            v_hist = v_cache_storage[slot_mapping, :max_l].to(q_nope.dtype)
            if target_heads_k != self.num_heads: k_hist = k_hist.repeat_interleave(self.num_heads // target_heads_k, dim=2)
            if target_heads_v != self.num_heads: v_hist = v_hist.repeat_interleave(self.num_heads // target_heads_v, dim=2)
            
            # FORCE EAGER MODE FOR NON-8 MULTIPLES OR NON-64 COMPATIBLE DIMS
            hd = q_nope.shape[-1]
            if True: # Always use Eager for MLA to ensure 100% stability across BS/Ctx
                pad_hd = (hd + 7) // 8 * 8
                q_p = F.pad(q_nope, (0, pad_hd - hd)).transpose(1, 2).reshape(-1, seq_len, pad_hd)
                k_p = F.pad(k_hist, (0, pad_hd - hd)).transpose(1, 2).reshape(-1, max_l, pad_hd)
                v_p = F.pad(v_hist, (0, pad_hd - hd)).transpose(1, 2).reshape(-1, max_l, pad_hd)
                
                scores = torch.matmul(q_p, k_p.transpose(-1, -2)) * self.scale
                mask = torch.triu(torch.ones((seq_len, max_l), device=q.device), diagonal=kv_start_indices[0].item() + 1).bool()
                scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
                attn = F.softmax(scores, dim=-1)
                output = torch.matmul(attn, v_p).view(bsz, self.num_heads, seq_len, pad_hd).transpose(1, 2)[..., :hd]
            else:
                output = F.scaled_dot_product_attention(q_nope.transpose(1, 2), k_hist.transpose(1, 2), v_hist.transpose(1, 2), is_causal=True, scale=self.scale).transpose(1, 2)
            output = output.reshape(bsz, seq_len, -1)
        else:
            from vllm.kernels.triton.paged_attention import paged_attention_v1
            max_len = k_cache_storage.shape[1]; block_tables = slot_mapping.view(bsz, 1).to(torch.int32)
            output = torch.empty((bsz, self.num_heads, target_dim_v), device=q.device, dtype=q.dtype)
            paged_attention_v1(output, q_nope.view(bsz, self.num_heads, actual_q_nope_dim), k_cache_storage, v_cache_storage, target_heads_k, self.scale, block_tables, seq_lens, block_size=max_len, max_seq_len=max_len, alibi_slopes=None, kv_cache_dtype="fp8" if k_cache_storage.dtype == torch.float8_e4m3fn else "fp16", k_scale=None, v_scale=None)
            output = output.view(bsz, 1, -1)
        return self.o_proj(output, lora_mapping=lora_mapping)

class DeepSeekV2MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config; self.num_experts = config.n_routed_experts; self.topk = config.num_experts_per_tok
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_exps", device="cpu")
        self.w1_up = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up_exps", device="cpu")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_exps", device="cpu")
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False).cuda().half()
        self._expert_triplet_cache = OrderedDict(); self._max_expert_cache_size = 64

    def _get_expert_triplet(self, expert_id, dtype):
        cache_key = (expert_id, dtype)
        if cache_key in self._expert_triplet_cache: return self._expert_triplet_cache[cache_key]
        inter_size = self.config.moe_intermediate_size; h_size = self.config.hidden_size
        w1 = self.w1.weight[expert_id * inter_size : (expert_id + 1) * inter_size].to("cuda").to(dtype)
        up = self.w1_up.weight[expert_id * inter_size : (expert_id + 1) * inter_size].to("cuda").to(dtype)
        w2 = self.w2.weight[expert_id * h_size : (expert_id + 1) * h_size].to("cuda").to(dtype)
        if w2.shape[0] != self.config.hidden_size: w2 = w2[:self.config.hidden_size].contiguous()
        self._expert_triplet_cache[cache_key] = (w1, up, w2)
        if len(self._expert_triplet_cache) > self._max_expert_cache_size: self._expert_triplet_cache.popitem(last=False)
        return (w1, up, w2)

    def forward(self, x, lora_mapping=None):
        x = x.half(); orig_shape = x.shape; x = x.view(-1, x.shape[-1])
        logits = self.gate(x); topk_weights, topk_ids = torch.topk(logits, self.topk, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1); out = torch.zeros_like(x)
        unique_experts = torch.unique(topk_ids)
        for expert_id in unique_experts:
            eid = expert_id.item(); mask = (topk_ids == eid); token_indices, topk_indices = torch.where(mask)
            if token_indices.numel() == 0: continue
            w1, up, w2 = self._get_expert_triplet(eid, torch.float16)
            tokens = x[token_indices]; h = F.silu(tokens @ w1.T) * (tokens @ up.T); expert_out = (h @ w2.T)
            if expert_out.dim() == 1: expert_out = expert_out.unsqueeze(0)
            if expert_out.shape[-1] != x.shape[-1]:
                t = x.shape[-1]
                if expert_out.shape[-1] > t: expert_out = expert_out[..., :t]
                else: expert_out = F.pad(expert_out, (0, t - expert_out.shape[-1]))
            weights = topk_weights[token_indices, topk_indices].unsqueeze(-1).half()
            out.index_add_(0, token_indices, expert_out * weights)
        return out.view(orig_shape)

class DeepSeekV2MLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        g = self.gate_proj(x, lora_mapping=lora_mapping); u = self.up_proj(x, lora_mapping=lora_mapping)
        return self.down_proj(self.act(g) * u, lora_mapping=lora_mapping)

class DeepSeekV2Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekV2Attention(config, layer_id, quant_config, prefix)
        if config.n_routed_experts > 0 and layer_id >= config.first_k_dense_replace: self.mlp = DeepSeekV2MoE(config, quant_config, prefix)
        else: self.mlp = DeepSeekV2MLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(hidden_states)
        attn_res = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        hidden_states = hidden_states + attn_res
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping)
        return hidden_states

class DeepSeekV2Model(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV2Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        if self.embed_tokens.weight.device != input_ids.device: x = self.embed_tokens(input_ids.cpu()).to(input_ids.device)
        else: x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(x)

class DeepSeekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = DeepSeekV2Model(vllm_config, prefix)
        self.lm_head = LiteLinear(getattr(vllm_config.model_config.hf_config, "hidden_size", 4096), getattr(vllm_config.model_config.hf_config, "vocab_size", 151936), bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        last_hidden = hidden_states[:, -1:, :]
        return self.lm_head(last_hidden, lora_mapping=lora_mapping)

DeepseekV2ForCausalLM = DeepSeekV2ForCausalLM
