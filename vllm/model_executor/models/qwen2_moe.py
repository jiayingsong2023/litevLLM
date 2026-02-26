# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.models.lite_base import LiteModel, LiteDecoderLayer
from vllm.attention.backends.triton_attn import TritonAttention

class Qwen2MoeMLP(nn.Module):
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

class Qwen2MoeSparseMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = LiteLinear(config.hidden_size, config.num_experts, bias=False,
                               quant_config=quant_config, prefix=f"{prefix}.gate")
        
        # Experts are stored as fused weights for Triton kernel
        self.experts_w1 = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size))
        self.experts_w2 = nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        return fused_moe(hidden_states, self.experts_w1, self.experts_w2, router_logits, 
                         topk=self.config.num_experts_per_tok)

class Qwen2MoeLayer(LiteDecoderLayer):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.self_attn = TritonAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        
        # Qwen2-MoE has shared experts and sparse experts
        if config.num_experts > 0:
            self.mlp = Qwen2MoeSparseMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen2MoeMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class Qwen2MoeModel(LiteModel):
    def __init__(self, vllm_config, prefix=""):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2MoeLayer(config, i, quant_config=vllm_config.quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class Qwen2MoeForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen2MoeModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size,
                                 vllm_config.model_config.hf_config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states)