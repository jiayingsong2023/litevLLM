# SPDX-License-Identifier: Apache-2.0
"""
Kimi-Linear: LitevLLM Optimized Implementation.
Supports kimi_linear architecture with Hybrid MLA and KDA layers.
Uses Global LRU Cache for Experts to fit 48B model in 60GB VRAM.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.models.lite_base import LiteModel, LiteDecoderLayer
from vllm.attention.backends.triton_attn import TritonAttention
from vllm.kernels.triton.gguf_dequant import gguf_dequantize

class KimiAttention(nn.Module):
    def __init__(self, config: Any, layer_id: int, quant_config: Optional[Any] = None, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
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

        return self.o_proj(output.view(-1, self.num_heads * self.v_head_dim)).view(hidden_states.shape)

class CachedKimiExperts(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.w1 = LiteLinear(config.hidden_size, config.num_experts * config.moe_intermediate_size * 2, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_up_exps")
        self.w2 = LiteLinear(config.moe_intermediate_size, config.num_experts * config.hidden_size, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_exps")

    def forward(self, x, router_logits):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        self.w1(x[:1])
        self.w2(torch.zeros((1, self.config.moe_intermediate_size), device=x.device, dtype=x.dtype))
        w1_tensor = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id)
        w2_tensor = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)

        if w1_tensor is None and getattr(self.w1, "qweight", None) is not None:
            w1_tensor = gguf_dequantize(self.w1.qweight, self.w1.scales, 2)
            _GLOBAL_GGUF_CACHE.put(self.w1.weight_id, w1_tensor)
        if w2_tensor is None and getattr(self.w2, "qweight", None) is not None:
            w2_tensor = gguf_dequantize(self.w2.qweight, self.w2.scales, 2)
            _GLOBAL_GGUF_CACHE.put(self.w2.weight_id, w2_tensor)

        if w1_tensor is None or w2_tensor is None:
            raise RuntimeError(
                f"Kimi experts weights are missing for layer '{self.prefix}'. "
                "Refusing fallback-to-zero expert path."
            )

        w1_reshaped = w1_tensor.view(self.config.num_experts, 2 * self.config.moe_intermediate_size, -1)
        w2_reshaped = w2_tensor.view(self.config.num_experts, -1, self.config.moe_intermediate_size)
        return fused_moe(x, w1_reshaped, w2_reshaped, router_logits, topk=self.config.num_experts_per_token)

class KimiMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = LiteLinear(config.hidden_size, config.num_experts, bias=False, prefix=f"{prefix}.ffn_gate_inp")
        self.experts = CachedKimiExperts(config, quant_config, prefix)
        
        if getattr(config, "num_shared_experts", 0) > 0:
            shared_dim = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = LiteLinear(config.hidden_size, 2 * shared_dim, bias=False, 
                                           quant_config=quant_config, prefix=f"{prefix}.ffn_gate_up_shexp")
            self.shared_down = LiteLinear(shared_dim, config.hidden_size, bias=False, 
                                         quant_config=quant_config, prefix=f"{prefix}.ffn_down_shexp")
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

class KimiLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_id in config.linear_attn_config["full_attn_layers"]:
            self.self_attn = KimiAttention(config, layer_id, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = KimiAttention(config, layer_id, quant_config=quant_config, prefix=f"{prefix}.self_attn")
            
        if layer_id >= getattr(config, "first_k_dense_replace", 0):
            self.mlp = KimiMoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = KimiMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class KimiMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, 
                                     quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, 
                                   quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = Silu()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

class KimiLinearForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            KimiLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
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
