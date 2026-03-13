# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from .lite_config import LiteConfig

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        
        # Flattened for Tiling visibility
        self.q_proj = LiteLinear(config.hidden_size, config.num_attention_heads * 128, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj")
        self.kv_a_proj = LiteLinear(config.hidden_size, 512, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_a_proj")
        self.kv_b_proj = LiteLinear(512, config.num_attention_heads * 128, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_b_proj")
        self.o_proj = LiteLinear(config.num_attention_heads * 128, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")

    def forward(self, x, positions, kv_cache, attn_metadata):
        if hasattr(self, "_fast_forward"): return self._fast_forward(x, positions, kv_cache, attn_metadata)
        h = self.input_layernorm(x)
        return self.o_proj(self.q_proj(h)) + x

    def compile_fast_dispatch(self):
        from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
        from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant
        TILE_SIZE = 1024
        
        def _get_op(proj):
            q_type, d = proj.get_fast_data()
            if q_type == "fp16": return lambda x: F.linear(x, d[0])
            if q_type == "awq":
                tiles = [(d[0][:, s//8:min(s+TILE_SIZE, proj.input_size)//8].contiguous(), d[1][:, s//128:min(s+TILE_SIZE, proj.input_size)//128].contiguous(), d[2][:, s//128//8:min(s+TILE_SIZE, proj.input_size)//128//8+1].contiguous() if d[2] is not None else None, d[3], s, min(s+TILE_SIZE, proj.input_size)) for s in range(0, proj.input_size, TILE_SIZE)]
                return lambda x: sum(awq_fused_gemm(x[:, t[4]:t[5]].contiguous(), *t[:4]) for t in tiles)
            if q_type == "gguf":
                tiles = [(d[0][:, (s//32)*18 : min(s+TILE_SIZE, proj.input_size)//32*18].contiguous(), proj.output_size, min(s+TILE_SIZE, proj.input_size)-s, s, min(s+TILE_SIZE, proj.input_size)) for s in range(0, proj.input_size, TILE_SIZE)]
                return lambda x: sum(F.linear(x[:, t[3]:t[4]].contiguous(), gguf_q4_0_dequant(t[0], t[1], t[2])) for t in tiles)
            return lambda x: F.linear(x, d[0])

        self._q_op = _get_op(self.q_proj); self._o_op = _get_op(self.o_proj)
        self._gate_op = _get_op(self.gate_proj); self._up_op = _get_op(self.up_proj); self._down_op = _get_op(self.down_proj)
        in_w = self.input_layernorm.weight; post_w = self.post_attention_layernorm.weight; eps = self.input_layernorm.variance_epsilon

        @torch.inference_mode()
        def _fast_forward(hidden_states, positions, kv_cache, attn_metadata):
            h_norm = F.rms_norm(hidden_states.view(-1, hidden_states.shape[-1]), (hidden_states.shape[-1],), in_w, eps)
            q = self._q_op(h_norm)
            # Placeholder for MLA Logic - Focus on GEMM Tiling Stability for now
            hidden_states = hidden_states + self._o_op(q).view(hidden_states.shape)
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            h_norm_post = F.rms_norm(h_flat, (h_flat.shape[-1],), post_w, eps)
            mlp_out = self._down_op(F.silu(self._gate_op(h_norm_post)) * self._up_op(h_norm_post))
            return hidden_states + mlp_out.view(hidden_states.shape)
        self._fast_forward = _fast_forward

class DeepseekV2Model(nn.Module):
    def __init__(self, hf_config, quant_config):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(self.config, quant_config, f"model.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = DeepseekV2Model(vllm_config.model_config.hf_config, vllm_config.quant_config)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = torch.ones((input_ids.shape[0], self.model.config.hidden_size), device=input_ids.device, dtype=torch.float16)
        for i in range(len(self.model.layers)): x = self.model.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.model.norm(x)[:, -1:, :]
