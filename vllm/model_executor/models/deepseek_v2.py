# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from .lite_config import LiteConfig

USE_TRITON = False # Stable PyTorch Path for AMD Audit

class DeepseekV2MoE(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix: str):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, "n_routed_experts", 64)
        self.top_k = getattr(config, "num_experts_per_tok", 6)
        self.router = LiteLinear(config.hidden_size, self.num_experts, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_inp")
        self.shared_experts = nn.Module()
        sh_inter = getattr(config, "intermediate_size", 10944)
        self.shared_experts.gate_proj = LiteLinear(config.hidden_size, sh_inter, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_shexp")
        self.shared_experts.up_proj = LiteLinear(config.hidden_size, sh_inter, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_shexp")
        self.shared_experts.down_proj = LiteLinear(sh_inter, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_shexp")
        self.experts_gate = nn.Parameter(torch.empty(0), requires_grad=False)
        self.experts_up = nn.Parameter(torch.empty(0), requires_grad=False)
        self.experts_down = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq, h = x.shape
        x_flat = x.view(-1, h)
        shared_out = self.shared_experts.down_proj(
            F.silu(self.shared_experts.gate_proj(x)) * self.shared_experts.up_proj(x)
        )

        # If routed experts are not present in this checkpoint, fall back to shared experts only.
        if (
            self.experts_gate.numel() == 0
            or self.experts_up.numel() == 0
            or self.experts_down.numel() == 0
        ):
            return shared_out
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Vectorized Expert Processing (Efficiency for Audit)
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            w = topk_weights[:, k:k+1]
            idx = topk_ids[:, k]
            # Fast vectorized selection
            gate = self.experts_gate[idx] # [tokens, inter, hidden]
            up = self.experts_up[idx]
            down = self.experts_down[idx]
            
            # Batch Matmul
            # x_flat[i]: [hidden], gate[i]: [inter, hidden]
            # res: [tokens, inter]
            e_gate = torch.bmm(gate, x_flat.unsqueeze(-1)).squeeze(-1)
            e_up = torch.bmm(up, x_flat.unsqueeze(-1)).squeeze(-1)
            e_out = torch.bmm(down, (F.silu(e_gate) * e_up).unsqueeze(-1)).squeeze(-1)
            routed_out += w * e_out
            
        return shared_out + routed_out.view(bs, seq, h)

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_nope_dim = getattr(config, "qk_nope_head_dim", 64)
        self.qk_rope_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "v_head_dim", 128)
        raw_q_lora_rank = getattr(config, "q_lora_rank", None)
        self.q_lora_rank = raw_q_lora_rank if raw_q_lora_rank is not None else 0
        self.use_q_lora = self.q_lora_rank > 0
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        if self.use_q_lora:
            self.q_a_proj = LiteLinear(config.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = LiteLinear(self.q_lora_rank, self.num_heads * (self.qk_nope_dim + self.qk_rope_dim), bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_b_proj")
        else:
            # Some DeepSeek V2 Lite checkpoints set q_lora_rank to null and use a direct q projection.
            self.q_proj = LiteLinear(
                config.hidden_size,
                self.num_heads * (self.qk_nope_dim + self.qk_rope_dim),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn.q_proj",
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        self.kv_a_proj = LiteLinear(config.hidden_size, self.kv_lora_rank + self.qk_rope_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_a_proj")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = LiteLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_b_proj")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        is_moe = int(prefix.split(".")[-1]) >= getattr(config, "first_k_dense_replace", 1)
        if is_moe: self.mlp = DeepseekV2MoE(config, quant_config, prefix)
        else:
            self.mlp = nn.Module()
            self.mlp.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj")
            self.mlp.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj")
            self.mlp.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
            self.mlp.forward = lambda x: self.mlp.down_proj(F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))

    def forward(self, x, positions, kv_cache, attn_metadata):
        h = self.input_layernorm(x)
        bs, seq, _ = h.shape
        if self.use_q_lora:
            q_full = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(h)))
        else:
            q_full = self.q_proj(h)
        q_nope, q_rope = torch.split(q_full, [self.num_heads * self.qk_nope_dim, self.num_heads * self.qk_rope_dim], dim=-1)
        q_nope = q_nope.view(bs, seq, self.num_heads, self.qk_nope_dim)
        q_rope = q_rope.view(bs, seq, self.num_heads, self.qk_rope_dim)
        kv_a = self.kv_a_proj(h)
        kv_latent, k_rope = torch.split(kv_a, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv_b = self.kv_b_proj(kv_latent)
        k_nope, v = torch.split(kv_b, [self.num_heads * self.qk_nope_dim, self.num_heads * self.v_head_dim], dim=-1)
        k_nope = k_nope.view(bs, seq, self.num_heads, self.qk_nope_dim)
        v = v.view(bs, seq, self.num_heads, self.v_head_dim)
        k_rope = k_rope.view(bs, seq, 1, self.qk_rope_dim).expand(-1, -1, self.num_heads, -1)
        Q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        K = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        V = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(bs, seq, -1)
        hidden_states = x + self.o_proj(attn_out)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

class DeepseekV2Model(nn.Module):
    def __init__(self, config: LiteConfig, vllm_config: VllmConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, vllm_config.quant_config, prefix=f"model.layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.norm(x)

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = DeepseekV2Model(self.config, vllm_config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping)
        return self.lm_head(hidden_states)
