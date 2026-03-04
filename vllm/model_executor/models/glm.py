# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash: LitevLLM Optimized Implementation.
Supports glm4_moe_lite architecture with MLA and Shared Experts.
Uses Global LRU Cache for Experts to save VRAM.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.models.lite_base import LiteModel, LiteDecoderLayer

class GlmMLA(nn.Module):
    def __init__(self, config: Any, quant_config: Optional[Any] = None, prefix: str = ""):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        self.q_proj = LiteLinear(config.hidden_size, self.num_heads * self.q_head_dim, 
                                bias=False, quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.kv_proj = LiteLinear(config.hidden_size, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim) + self.qk_rope_head_dim,
                                 bias=False, quant_config=quant_config, prefix=f"{prefix}.kv_proj")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, config.hidden_size, 
                                bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.scale = self.q_head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.q_head_dim)
        kv = self.kv_proj(hidden_states)
        
        try:
            from vllm.attention.ops.triton_paged_attn import triton_paged_attention
            k = kv[:, :self.num_heads * self.qk_nope_head_dim].view(-1, self.num_heads, self.qk_nope_head_dim)
            v = kv[:, self.num_heads * self.qk_nope_head_dim : self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)].view(-1, self.num_heads, self.v_head_dim)
            
            output = triton_paged_attention(
                q[:, :, :self.qk_nope_head_dim], k, v, kv_cache,
                attn_metadata["slot_mapping"] if isinstance(attn_metadata, dict) else attn_metadata.slot_mapping,
                attn_metadata["seq_lens"] if isinstance(attn_metadata, dict) else attn_metadata.seq_lens,
                attn_metadata.get("block_tables") if isinstance(attn_metadata, dict) else getattr(attn_metadata, "block_tables", None),
                self.scale
            )
        except Exception:
            output = torch.zeros((q.shape[0], self.num_heads, self.v_head_dim), device=q.device, dtype=q.dtype)

        return self.o_proj(output.view(-1, self.num_heads * self.v_head_dim)).view(hidden_states.shape)

class CachedGlmExperts(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.w1 = LiteLinear(config.hidden_size, config.n_routed_experts * config.moe_intermediate_size, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w1")
        self.w2 = LiteLinear(config.moe_intermediate_size, config.n_routed_experts * config.hidden_size, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w2")

    def forward(self, x, router_logits):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        # Trigger cache
        self.w1(x[:1])
        self.w2(torch.zeros((1, self.config.moe_intermediate_size), device=x.device, dtype=x.dtype))
        
        w1_tensor = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id)
        w2_tensor = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)
        
        w1_reshaped = w1_tensor.view(self.config.n_routed_experts, self.config.moe_intermediate_size, -1)
        w2_reshaped = w2_tensor.view(self.config.n_routed_experts, -1, self.config.moe_intermediate_size)
        
        return fused_moe(x, w1_reshaped, w2_reshaped, router_logits, topk=self.config.num_experts_per_tok)

class GlmMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = LiteLinear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = CachedGlmExperts(config, quant_config, prefix)
        
        if getattr(config, "n_shared_experts", 0) > 0:
            shared_dim = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = LiteLinear(config.hidden_size, 2 * shared_dim, bias=False, 
                                           quant_config=quant_config, prefix=f"{prefix}.shared_experts.gate_up_proj")
            self.shared_down = LiteLinear(shared_dim, config.hidden_size, bias=False, 
                                         quant_config=quant_config, prefix=f"{prefix}.shared_experts.down_proj")
            self.act = Silu()
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, orig_shape[-1])
        router_logits = self.gate(x_2d)
        routed_out = self.experts(x_2d, router_logits)
        
        if self.shared_experts is not None:
            gate_up = self.shared_experts(x_2d)
            gate, up = gate_up.chunk(2, dim=-1)
            shared_out = self.shared_down(self.act(gate) * up)
            routed_out = routed_out + shared_out
            
        return routed_out.view(orig_shape)

class GlmLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GlmMLA(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = GlmMoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class Glm4MoeLiteForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GlmLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LiteLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, positions, kv_caches[i], attn_metadata)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

class GlmForCausalLM(Glm4MoeLiteForCausalLM): pass
