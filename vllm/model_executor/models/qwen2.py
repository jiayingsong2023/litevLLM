# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Qwen2 model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Set, Tuple, Any
from transformers import Qwen2Config

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.compilation.decorators import support_torch_compile

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config=None, params_dtype=torch.float16, prefix=""):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, params_dtype=params_dtype, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config=None, params_dtype=torch.float16, prefix=""):
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

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config=None, params_dtype=torch.float16, prefix=""):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, quant_config, params_dtype, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config, quant_config, params_dtype, prefix=f"{prefix}.mlp")

    def forward(self, positions, x, **kwargs):
        x = x + self.self_attn(positions, self.input_layernorm(x), **kwargs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class Qwen2Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        params_dtype = vllm_config.model_config.dtype
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, vllm_config.quant_config, params_dtype, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(positions, x, **kwargs)
        return self.norm(x)

class Qwen2ForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsLoRA):
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
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=f"{prefix}.model")
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
                        # Use LiteLinear's weight_loader
                        param.weight_loader(param, (loaded_weight, shard_id))
                        loaded_params.add(new_name)
                        break
            else:
                # Handle normal parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader:
                        weight_loader(param, loaded_weight)
                    else:
                        param.data.copy_(loaded_weight)
                    loaded_params.add(name)

        return loaded_params
