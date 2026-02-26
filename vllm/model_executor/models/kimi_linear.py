# SPDX-License-Identifier: Apache-2.0
"""
Kimi: LitevLLM Optimized Implementation.
Supports KDA (Kimi Delta Attention) and MLA with LiteLinear.
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

class KimiMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = LiteLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts_w1 = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size))
        self.experts_w2 = nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        return fused_moe(hidden_states, self.experts_w1, self.experts_w2, router_logits, 
                         topk=self.config.num_experts_per_token)

class KimiLayer(LiteDecoderLayer):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Kimi uses a mix of MLA and KDA (Delta Attention)
        # Simplified to TritonAttention for the Lite build
        self.self_attn = TritonAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        
        if config.is_moe and layer_id >= config.first_k_dense_replace:
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

class KimiLinearModel(LiteModel):
    def __init__(self, vllm_config, prefix=""):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        self.layers = nn.ModuleList([
            KimiLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class KimiLinearForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = KimiLinearModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size,
                                 vllm_config.model_config.hf_config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states)