# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Qwen2Moe model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Set, Tuple, Any, Callable
import typing
from transformers import Qwen2MoeConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.compilation.decorators import support_torch_compile

class Qwen2MoeMLP(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, intermediate_size: int, quant_config=None, params_dtype=torch.float16, prefix="", expert_gate=None):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * intermediate_size, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LiteLinear(intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        out = self.down_proj(self.act_fn(self.gate_up_proj(x)))
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)) * out
        return out

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, quant_config=None, params_dtype=torch.float16, prefix=""):
        super().__init__()
        self.gate = LiteLinear(config.hidden_size, config.num_experts, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.gate")
        self.shared_expert_gate = LiteLinear(config.hidden_size, 1, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.shared_expert_gate")
        
        self.shared_expert = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(config, config.shared_expert_intermediate_size, quant_config, params_dtype, f"{prefix}.shared_expert", self.shared_expert_gate)

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.experts"
        )

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        shared_out, fused_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if shared_out is not None:
            return shared_out + fused_out
        return fused_out

class Qwen2MoeAttention(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, quant_config=None, params_dtype=torch.float16, prefix=""):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        
        self.qkv_proj = LiteLinear(config.hidden_size, (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim, bias=True, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.qkv_proj")
        self.o_proj = LiteLinear(config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.o_proj")
        
        self.attn = Attention(
            self.total_num_heads, self.head_dim, self.head_dim**-0.5,
            self.total_num_kv_heads, quant_config=quant_config,
            prefix=f"{prefix}.attn"
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            is_neox_style=True,
            rope_parameters=getattr(config, "rope_parameters", None),
            dtype=params_dtype,
        )

    def forward(self, positions, hidden_states, **kwargs):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            self.total_num_heads * self.head_dim,
            self.total_num_kv_heads * self.head_dim,
            self.total_num_kv_heads * self.head_dim
        ], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))

class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, quant_config=None, params_dtype=torch.float16, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2MoeAttention(config, quant_config, params_dtype, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MoeSparseMoeBlock(config, quant_config, params_dtype, prefix=f"{prefix}.mlp")

    def forward(self, positions, x, **kwargs):
        x = x + self.self_attn(positions, self.input_layernorm(x), **kwargs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class Qwen2MoeModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        params_dtype = vllm_config.model_config.dtype
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2MoeDecoderLayer(config, vllm_config.quant_config, params_dtype, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(positions, x, **kwargs)
        return self.norm(x)

class Qwen2MoeForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = Qwen2MoeModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]
        
        # Expert mapping for FusedMoE
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts
        )

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, loaded_weight in weights:
            # 1. Handle expert mapping
            is_expert_weight = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name in name:
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped in params_dict:
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        weight_loader(param, loaded_weight, name_mapped, shard_id, expert_id)
                        loaded_params.add(name_mapped)
                        break
            if is_expert_weight:
                continue

            # 2. Handle stacked parameters (QKV, Gate-Up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    new_name = name.replace(weight_name, param_name)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        # Use LiteLinear's weight_loader
                        param.weight_loader(param, (loaded_weight, shard_id))
                        loaded_params.add(new_name)
                        break
            else:
                # 3. Handle normal parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader:
                        weight_loader(param, loaded_weight)
                    else:
                        param.data.copy_(loaded_weight)
                    loaded_params.add(name)

        return loaded_params