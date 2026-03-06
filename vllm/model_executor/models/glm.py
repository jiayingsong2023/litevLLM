# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.gguf_kernels import ggml_dequantize_fallback

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
        # This path only uses NOPE dimensions in dot-product attention.
        self.scale = self.qk_nope_head_dim**-0.5

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
        self.config = config
        self.num_experts = config.n_routed_experts
        self.w1 = LiteLinear(
            config.hidden_size,
            self.num_experts * config.moe_intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn_gate_exps",
        )
        self.w1_up = LiteLinear(
            config.hidden_size,
            self.num_experts * config.moe_intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn_up_exps",
        )
        self.w2 = LiteLinear(
            config.moe_intermediate_size,
            self.num_experts * config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn_down_exps",
        )

    def _dequantize_expert_matrix(self, layer: LiteLinear, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
        if getattr(layer, "qweight", None) is None:
            raise RuntimeError(f"GLM expert tensor not loaded for expert={expert_idx}")
        if getattr(layer, "gguf_quant_type", None) is None or getattr(layer, "gguf_shape", None) is None:
            raise RuntimeError(f"GLM expert quant metadata missing for expert={expert_idx}")
        packed = layer.qweight[expert_idx].contiguous()
        m_dim = int(layer.gguf_shape[1])
        n_dim = int(layer.gguf_shape[0])
        return ggml_dequantize_fallback(
            packed,
            int(layer.gguf_quant_type),
            m_dim,
            n_dim,
            dtype,
        )

    def forward(self, x, router_logits):
        if x.dim() == 2: x = x.unsqueeze(1)
        curr_x = x[:, -1:, :].view(-1, x.shape[-1])
        topk_vals, topk_ids = torch.topk(router_logits, k=self.config.num_experts_per_tok, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        output = torch.zeros_like(curr_x)
        for token_idx in range(curr_x.shape[0]):
            hidden_vec = curr_x[token_idx]
            token_out = torch.zeros_like(hidden_vec)
            for route_idx, expert_id in enumerate(topk_ids[token_idx]):
                expert_idx = int(expert_id.item())
                gate_w = self._dequantize_expert_matrix(self.w1, expert_idx, curr_x.dtype)
                up_w = self._dequantize_expert_matrix(self.w1_up, expert_idx, curr_x.dtype)
                down_w = self._dequantize_expert_matrix(self.w2, expert_idx, curr_x.dtype)
                gate_act = torch.mv(gate_w, hidden_vec)
                up_act = torch.mv(up_w, hidden_vec)
                mixed = F.silu(gate_act) * up_act
                expert_out = torch.mv(down_w, mixed)
                token_out = token_out + expert_out * topk_weights[token_idx, route_idx]
            output[token_idx] = token_out
        return output.view(x.shape[0], 1, -1)

class GlmMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config; self.gate = LiteLinear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = CachedGlmExperts(config, quant_config, prefix)
        if getattr(config, "n_shared_experts", 0) > 0:
            shared_dim = config.moe_intermediate_size * config.n_shared_experts
            self.shared_gate = LiteLinear(config.hidden_size, shared_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_shexp")
            self.shared_up = LiteLinear(config.hidden_size, shared_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up_shexp")
            self.shared_down = LiteLinear(shared_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_shexp")
            self.act = Silu()
        else:
            self.shared_gate = None
            self.shared_up = None
    def forward(self, hidden_states):
        x_2d = hidden_states[:, -1:, :].view(-1, hidden_states.shape[-1])
        routed_out = self.experts(hidden_states, self.gate(x_2d))
        if self.shared_gate is not None and self.shared_up is not None:
            g = self.shared_gate(x_2d)
            u = self.shared_up(x_2d)
            routed_out = routed_out + self.shared_down(self.act(g) * u).view(hidden_states.shape[0], 1, -1)
        return routed_out


class GlmMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down")
        self.act = Silu()
    def forward(self, x):
        x_2d = x[:, -1:, :]
        g = self.gate_proj(x_2d)
        u = self.up_proj(x_2d)
        return self.down_proj(self.act(g) * u)

class GlmLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GlmMLA(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        if layer_id >= getattr(config, "first_k_dense_replace", 0):
            self.mlp = GlmMoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = GlmMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
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
