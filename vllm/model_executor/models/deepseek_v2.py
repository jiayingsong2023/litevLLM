# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe

class DeepSeekV2Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.config = config; self.hidden_size = config.hidden_size; self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "v_head_dim", getattr(config, "head_dim", 128))
        self.kv_lora_rank = config.kv_lora_rank; self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.q_size = self.num_heads * self.head_dim; self.kv_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.qkv_proj = LiteLinear(self.hidden_size, self.q_size + self.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.qkv_proj")
        self.o_proj = LiteLinear(self.q_size, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, lora_mapping=lora_mapping)
        if qkv.dim() == 2: qkv = qkv.unsqueeze(1)
        q, kv = qkv.split([self.q_size, self.kv_size], dim=-1)
        from vllm.attention.ops.triton_paged_attn import triton_paged_attention
        output = triton_paged_attention(
            q[:, -1:, :].view(bsz, self.num_heads, self.head_dim),
            kv[:, -1:, :self.head_dim].view(bsz, 1, self.head_dim), 
            kv[:, -1:, :self.head_dim].view(bsz, 1, self.head_dim), 
            kv_cache, attn_metadata["slot_mapping"], attn_metadata["seq_lens"], None, self.scale
        )
        return self.o_proj(output.view(bsz, 1, -1), lora_mapping=lora_mapping)

class DeepSeekV2MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.num_experts = config.n_routed_experts; self.topk = config.num_experts_per_tok
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.experts_w1")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.experts_w2")
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False).cuda().half()
    def forward(self, x, lora_mapping=None):
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        if x.dim() == 2: x = x.unsqueeze(1)
        curr_x = x[:, -1:, :].view(-1, x.shape[-1])
        # Experts bypass LoRA for performance
        self.w1(curr_x[:1]); self.w2(torch.zeros((1, self.w2.input_size), device=x.device, dtype=x.dtype))
        w1_t = _GLOBAL_GGUF_CACHE.get(self.w1.weight_id); w2_t = _GLOBAL_GGUF_CACHE.get(self.w2.weight_id)
        if w1_t is None or w2_t is None: return x[:, -1:, :]
        return fused_moe(curr_x, w1_t.view(self.num_experts, -1, curr_x.shape[-1]), w2_t.view(self.num_experts, curr_x.shape[-1], -1), self.gate(curr_x), topk=self.topk).view(x.shape[0], 1, -1)

class DeepSeekV2Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekV2Attention(config, layer_id, quant_config, prefix)
        if config.n_routed_experts > 0 and layer_id >= config.first_k_dense_replace: self.mlp = DeepSeekV2MoE(config, quant_config, prefix)
        else: self.mlp = DeepSeekV2MLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(hidden_states)
        attn_res = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        if hidden_states.dim() == 3: hidden_states = hidden_states[:, -1:, :] + attn_res
        else: hidden_states = hidden_states + attn_res.squeeze(1)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping)
        return hidden_states

class DeepSeekV2MLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        g, u = self.gate_up_proj(x[:, -1:, :], lora_mapping=lora_mapping).chunk(2, dim=-1); return self.down_proj(self.act(g) * u, lora_mapping=lora_mapping)

class DeepSeekV2Model(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV2Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(x)

class DeepSeekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = DeepSeekV2Model(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        return self.lm_head(self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping), lora_mapping=lora_mapping)

DeepseekV2ForCausalLM = DeepSeekV2ForCausalLM
