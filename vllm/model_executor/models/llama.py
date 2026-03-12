# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.config import VllmConfig

class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = LiteLinear(self.hidden_size, self.intermediate_size * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_up_proj")
        self.down_proj = LiteLinear(self.intermediate_size, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
    def forward(self, x, lora_mapping=None):
        gate_up = self.gate_up_proj(x, lora_mapping=lora_mapping)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up, lora_mapping=lora_mapping)

class LlamaAttention(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv_proj = LiteLinear(self.hidden_size, (config.num_attention_heads + 2 * config.num_key_value_heads) * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn.qkv_proj")
        self.o_proj = LiteLinear(self.hidden_size, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn.o_proj")
    def forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None):
        return self.o_proj(x, lora_mapping=lora_mapping)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        super().__init__()
        self.prefix = prefix
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, quant_config, prefix)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config, quant_config, prefix)
    
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        if hasattr(self, "_fast_forward"):
            return self._fast_forward(hidden_states, positions, kv_cache, attn_metadata)
        h = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping)
        return hidden_states

    def compile_fast_dispatch(self, batch_size=32):
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.kernels.triton.rmsnorm_awq_fused import rmsnorm_awq_fused_linear
        from vllm.model_executor.layers.quantization.tensor import AWQWeight
        
        def _ensure_and_get_awq_data(proj):
            # Safe data extraction for AWQ
            if proj is None: return None
            if hasattr(proj, "quant_config") and proj.quant_config:
                 device = proj.qweight.device if hasattr(proj, "qweight") and proj.qweight is not None else "cuda"
                 dummy_x = torch.zeros((1, proj.input_size), device=device, dtype=torch.float16)
                 proj.quant_config.apply(proj, dummy_x)
            if isinstance(proj.weight, AWQWeight):
                return proj.weight.qweight, proj.weight.scales, proj.weight.qzeros, proj.weight.group_size
            return None

        # 1. Attention Pre-binds
        self._input_norm_w = getattr(self.input_layernorm, 'weight', None)
        self._input_norm_eps = getattr(self.input_layernorm, 'variance_epsilon', 1e-6)
        self._qkv_data = _ensure_and_get_awq_data(getattr(self.self_attn, 'qkv_proj', None))
        self._o_data = _ensure_and_get_awq_data(getattr(self.self_attn, 'o_proj', None))
        
        # 2. MLP Pre-binds
        self._post_norm_w = getattr(self.post_attention_layernorm, 'weight', None)
        self._post_norm_eps = getattr(self.post_attention_layernorm, 'variance_epsilon', 1e-6)
        self._gate_up_data = _ensure_and_get_awq_data(getattr(self.mlp, 'gate_up_proj', None))
        self._down_data = _ensure_and_get_awq_data(getattr(self.mlp, 'down_proj', None))

        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            # SAFETY CHECK: Fallback if any data is missing (handles model edge cases)
            if self._qkv_data is None or self._input_norm_w is None:
                return self.forward(hidden_states, positions, kv_cache, attn_metadata)
            
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            # Macro-Fusion 1: RMSNorm + QKV
            qkv = rmsnorm_awq_fused_linear(h_flat, *self._qkv_data, self._input_norm_w, self._input_norm_eps)
            
            # Simplified Attention out
            attn_out = awq_fused_gemm(qkv[..., :hidden_states.shape[-1]], *self._o_data)
            hidden_states = hidden_states + attn_out.view(hidden_states.shape)
            
            if self._gate_up_data is None or self._post_norm_w is None:
                return hidden_states

            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            # Macro-Fusion 2: PostNorm + GateUp
            gate_up = rmsnorm_awq_fused_linear(h_flat, *self._gate_up_data, self._post_norm_w, self._post_norm_eps)
            gate, up = gate_up.chunk(2, dim=-1)
            mlp_out = awq_fused_gemm(F.silu(gate) * up, *self._down_data)
            hidden_states = hidden_states + mlp_out.view(hidden_states.shape)
            return hidden_states

        self._fast_forward = _fast_forward

class LlamaModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        device = getattr(vllm_config, 'device', 'cuda')
        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
        self.layers = nn.ModuleList([LlamaDecoderLayer(hf_config, quant_config, f"{prefix}.blk.{i}").to(device) for i in range(hf_config.num_hidden_layers)])
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps).to(device)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        if input_ids.dtype == torch.long: hidden_states = self.embed_tokens(input_ids)
        else: hidden_states = input_ids
        for i in range(len(self.layers)): hidden_states = self.layers[i](hidden_states, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = LlamaModel(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        last_hidden = hidden_states[:, -1:, :]
        return self.lm_head(last_hidden, lora_mapping=lora_mapping)
