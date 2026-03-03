# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V2/V3: LitevLLM Optimized Implementation.
Standardized on LiteLinear and Single-GPU Triton kernels.
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

class DeepseekV2MLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, 
                                     quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, 
                                   quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = Silu()

    def forward(self, x, lora_mapping=None):
        gate_up = self.gate_up_proj(x, lora_mapping=lora_mapping)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up, lora_mapping=lora_mapping)

class DeepseekV2MoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = LiteLinear(config.hidden_size, config.n_routed_experts, bias=False)
        
        # 1. Routed Experts
        self.experts_w1 = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_intermediate_size, config.hidden_size))
        self.experts_w2 = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size, config.moe_intermediate_size))
        
        # 2. Shared Experts
        if getattr(config, "n_shared_experts", 0) > 0:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                type('obj', (object,), {
                    'hidden_size': config.hidden_size,
                    'intermediate_size': shared_intermediate
                }),
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts"
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        # Flatten to 2D for MoE processing
        orig_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, orig_shape[-1])
        
        router_logits = self.gate(x_2d)
        routed_out = fused_moe(x_2d, self.experts_w1, self.experts_w2, router_logits, 
                              topk=self.config.num_experts_per_tok)
        
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x_2d)
            routed_out = routed_out + shared_out
            
        return routed_out.view(orig_shape)

class DeepseekV2Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TritonAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        
        if layer_id >= getattr(config, "first_k_dense_replace", 0):
            self.mlp = DeepseekV2MoE(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = DeepseekV2MLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, DeepseekV2MoE):
             hidden_states = self.mlp(hidden_states)
        else:
             hidden_states = self.mlp(hidden_states, lora_mapping=lora_mapping)
             
        return residual + hidden_states

class DeepseekV2Model(LiteModel):
    def __init__(self, vllm_config, prefix=""):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV2Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(hidden_states)

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = DeepseekV2Model(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size,
                                 vllm_config.model_config.hf_config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        return self.lm_head(hidden_states, lora_mapping=lora_mapping)

class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): pass
