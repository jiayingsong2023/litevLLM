# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.attention.backends.triton_attn import TritonAttention

class CachedMoEExperts(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.w1 = LiteLinear(config.hidden_size, config.num_experts * config.moe_intermediate_size, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w1")
        self.w2 = LiteLinear(config.moe_intermediate_size, config.num_experts * config.hidden_size, 
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.experts_w2")

    def forward(self, x, router_logits):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        # Ensure weights are in cache via minimal dummy forward
        self.w1(x[:1]) 
        self.w2(torch.zeros((1, self.config.moe_intermediate_size), device=x.device, dtype=x.dtype))
        
        w1_tensor = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id)
        w2_tensor = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)
        
        # Reshape to standard MoE format [E, N, K]
        w1_reshaped = w1_tensor.view(self.config.num_experts, self.config.moe_intermediate_size, -1)
        w2_reshaped = w2_tensor.view(self.config.num_experts, -1, self.config.moe_intermediate_size)
        
        return fused_moe(x, w1_reshaped, w2_reshaped, router_logits, topk=self.config.num_experts_per_tok)

class Qwen3_5MoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False).cuda().half()
        self.experts = CachedMoEExperts(config, quant_config, prefix)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, orig_shape[-1])
        router_logits = self.gate(x_2d)
        out_2d = self.experts(x_2d, router_logits)
        return out_2d.view(orig_shape)

class Qwen3_5Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        text_config = getattr(config, "text_config", config)
        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.self_attn = TritonAttention(text_config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        if hasattr(text_config, "num_experts") and text_config.num_experts > 0:
            self.mlp = Qwen3_5MoE(text_config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = nn.Identity()

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class Qwen3_5Model(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}")
            for i in range(text_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, positions, kv_caches[i], attn_metadata)
        return self.norm(hidden_states)

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen3_5Model(vllm_config, prefix)
        text_config = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        self.lm_head = LiteLinear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states)

class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM): pass
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForCausalLM): pass
class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM): pass
