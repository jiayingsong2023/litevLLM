# SPDX-License-Identifier: Apache-2.0
"""
Base classes for LiteEngine models to reduce redundancy.
Standardizes Llama-like architectures (Llama, Qwen2, Mistral, etc.).
"""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Tuple, Any, List, Union

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import get_act_fn, get_act_and_mul_fn
from vllm.model_executor.layers.linear import LiteLinear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.compilation.decorators import support_torch_compile

class LiteMLP(nn.Module):
    """Generic MLP for Transformer models."""
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        hidden_act: str = "silu", 
        quant_config: Optional[Any] = None, 
        params_dtype: torch.dtype = torch.float16, 
        prefix: str = "",
        bias: bool = False
    ):
        super().__init__()
        self.gate_up_proj = LiteLinear(
            input_size=hidden_size, 
            output_size=[intermediate_size, intermediate_size], 
            bias=bias, 
            quant_config=quant_config, 
            params_dtype=params_dtype, 
            prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = LiteLinear(
            input_size=intermediate_size, 
            output_size=hidden_size, 
            bias=bias, 
            quant_config=quant_config, 
            params_dtype=params_dtype, 
            prefix=f"{prefix}.down_proj"
        )
        # Use act_and_mul for Llama-like GLU
        self.act_fn = get_act_and_mul_fn(hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LiteAttention(nn.Module):
    """Generic Attention for Transformer models."""
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: int, 
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        qkv_bias: bool = False,
        o_bias: bool = False,
        quant_config: Optional[Any] = None, 
        params_dtype: torch.dtype = torch.float16, 
        prefix: str = ""
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        
        self.qkv_proj = LiteLinear(
            input_size=hidden_size, 
            output_size=[num_heads * self.head_dim, num_kv_heads * self.head_dim, num_kv_heads * self.head_dim], 
            bias=qkv_bias, 
            quant_config=quant_config, 
            params_dtype=params_dtype, 
            prefix=f"{prefix}.qkv_proj"
        )
        self.o_proj = LiteLinear(
            input_size=hidden_size, 
            output_size=hidden_size, 
            bias=o_bias, 
            quant_config=quant_config, 
            params_dtype=params_dtype, 
            prefix=f"{prefix}.o_proj"
        )
        
        self.attn = Attention(
            num_heads=num_heads, 
            head_size=self.head_dim, 
            scale=self.head_dim**-0.5,
            num_kv_heads=num_kv_heads, 
            quant_config=quant_config,
            prefix=f"{prefix}.attn"
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_scaling,
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


class LiteMoE(nn.Module):
    """Generic MoE block using Triton FusedMoE."""
    def __init__(
        self,
        config: Any,
        quant_config: Optional[Any] = None,
        params_dtype: torch.dtype = torch.float16,
        prefix: str = ""
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.gate = LiteLinear(
            input_size=config.hidden_size,
            output_size=config.num_local_experts,
            bias=False,
            quant_config=None,
            params_dtype=params_dtype,
            prefix=f"{prefix}.gate"
        )
        
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class LiteDecoderLayer(nn.Module):
    """Generic Decoder Layer."""
    def __init__(
        self, 
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_idx: int = 0,
        config: Optional[Any] = None,
        quant_config: Optional[Any] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config or vllm_config.model_config.hf_config
        self.quant_config = quant_config or vllm_config.quant_config
        self.dtype = params_dtype or vllm_config.model_config.dtype
        
        self.input_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        rope_theta = getattr(self.config, "rope_theta", 10000.0)
        rope_scaling = getattr(self.config, "rope_scaling", None)
        max_position_embeddings = getattr(self.config, "max_position_embeddings", 8192)
        qkv_bias = getattr(self.config, "attention_bias", False)
        
        self.self_attn = LiteAttention(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            qkv_bias=qkv_bias,
            quant_config=self.quant_config,
            params_dtype=self.dtype,
            prefix=f"{prefix}.self_attn"
        )
        
        if getattr(self.config, "num_local_experts", 0) > 0:
            self.mlp = LiteMoE(
                config=self.config,
                quant_config=self.quant_config,
                params_dtype=self.dtype,
                prefix=f"{prefix}.block_sparse_moe"
            )
        else:
            self.mlp = LiteMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
                quant_config=self.quant_config,
                params_dtype=self.dtype,
                prefix=f"{prefix}.mlp",
                bias=getattr(self.config, "mlp_bias", False)
            )

    def forward(self, positions, x, **kwargs):
        x = x + self.self_attn(positions, self.input_layernorm(x), **kwargs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LiteMoEDecoderLayer(LiteDecoderLayer):
    """Helper for MoE models if needed explicitly."""
    pass


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class LiteModel(nn.Module):
    """Generic Transformer Model."""
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size
        )
        
        self.layers = nn.ModuleList([
            LiteDecoderLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{i}",
                layer_idx=i
            ) 
            for i in range(self.config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(positions, x, **kwargs)
        return self.norm(x)


class LiteForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsLoRA):
    """Generic Causal LM."""
    
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = LiteModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
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
        
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, loaded_weight in weights:
            # Handle stacked parameters (QKV, Gate-Up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    new_name = name.replace(weight_name, param_name)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        if hasattr(param, "weight_loader"):
                            param.weight_loader(param, (loaded_weight, shard_id))
                        loaded_params.add(new_name)
                        break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader:
                        weight_loader(param, loaded_weight)
                    else:
                        param.data.copy_(loaded_weight)
                    loaded_params.add(name)

        return loaded_params
