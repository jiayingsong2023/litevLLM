# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe

class GlmMLA(nn.Module):
    def __init__(self, config: Any, quant_config: Optional[Any] = None, prefix: str = ""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = LiteLinear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.kv_proj = LiteLinear(config.hidden_size, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim) + self.qk_rope_head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.kv_proj")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.scale = self.q_head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        bsz = hidden_states.shape[0]
        q = self.q_proj(hidden_states)
        kv = self.kv_proj(hidden_states)
        
        # Robust Dimension Normalization
        if q.dim() == 2: q = q.unsqueeze(1)
        if kv.dim() == 2: kv = kv.unsqueeze(1)
        
        from vllm.attention.ops.triton_paged_attn import triton_paged_attention
        k = kv[..., :self.num_heads * self.qk_nope_head_dim].view(bsz, self.num_heads, self.qk_nope_head_dim)
        v = kv[..., self.num_heads * self.qk_nope_head_dim : self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)].view(bsz, self.num_heads, self.v_head_dim)
        
        output = triton_paged_attention(
            q[:, -1:, :].view(bsz, self.num_heads, self.q_head_dim)[:, :, :self.qk_nope_head_dim], 
            k, v, kv_cache, attn_metadata["slot_mapping"], attn_metadata["seq_lens"], None, self.scale
        )
        return self.o_proj(output.view(bsz, 1, -1))

class CachedGlmExperts(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config; self.num_experts = config.n_routed_experts
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w1")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w2")
    def forward(self, x, router_logits):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        if x.dim() == 2: x = x.unsqueeze(1)
        curr_x = x[:, -1:, :].view(-1, x.shape[-1])
        self.w1(curr_x[:1]); self.w2(torch.zeros((1, self.w2.input_size), device=x.device, dtype=x.dtype))
        w1_t = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id); w2_t = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)
        if w1_t is None or w2_t is None: return x[:, -1:, :]
        return fused_moe(curr_x, w1_t.view(self.num_experts, -1, curr_x.shape[-1]), w2_t.view(self.num_experts, curr_x.shape[-1], -1), router_logits, topk=self.config.num_experts_per_tok).view(x.shape[0], 1, -1)

class GlmMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config; self.gate = LiteLinear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = CachedGlmExperts(config, quant_config, prefix)
        if getattr(config, "n_shared_experts", 0) > 0:
            shared_dim = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = LiteLinear(config.hidden_size, 2 * shared_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.shared_experts.gate_up_proj")
            self.shared_down = LiteLinear(shared_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.shared_experts.down_proj")
            self.act = Silu()
        else: self.shared_experts = None
    def forward(self, hidden_states):
        x_2d = hidden_states[:, -1:, :].view(-1, hidden_states.shape[-1])
        routed_out = self.experts(hidden_states, self.gate(x_2d))
        if self.shared_experts is not None:
            g, u = self.shared_experts(x_2d).chunk(2, dim=-1)
            routed_out = routed_out + self.shared_down(self.act(g) * u).view(hidden_states.shape[0], 1, -1)
        return routed_out

class GlmLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GlmMLA(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = GlmMoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        h = self.input_layernorm(hidden_states)
        attn_res = self.self_attn(h, positions, kv_cache, attn_metadata)
        if hidden_states.dim() == 3: hidden_states = hidden_states[:, -1:, :] + attn_res
        else: hidden_states = hidden_states + attn_res.squeeze(1)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

class Glm4MoeLiteForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GlmLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LiteLinear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.lm_head(self.norm(x))

class GlmForCausalLM(Glm4MoeLiteForCausalLM): pass
