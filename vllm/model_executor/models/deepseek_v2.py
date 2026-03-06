# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.gguf_kernels import ggml_dequantize_fallback

class DeepSeekV2Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.config = config; self.hidden_size = config.hidden_size; self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", getattr(config, "head_dim", 128))
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "v_head_dim", getattr(config, "head_dim", 128))
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank; self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.q_size = self.num_heads * self.q_head_dim; self.kv_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.q_proj = LiteLinear(self.hidden_size, self.q_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_q")
        self.kv_proj = LiteLinear(self.hidden_size, self.kv_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_kv_a_mqa")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn_output")
        self.scale = self.qk_nope_head_dim**-0.5

    def _match_dim(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        if tensor.shape[-1] == target_dim:
            return tensor
        if tensor.shape[-1] > target_dim:
            return tensor[..., : target_dim]
        return torch.nn.functional.pad(tensor, (0, target_dim - tensor.shape[-1]))

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz = hidden_states.shape[0]
        # Optimized path for decode (seqlen=1)
        hidden_last = hidden_states[:, -1:, :]
        q = self.q_proj(hidden_last, lora_mapping=lora_mapping).view(bsz, self.num_heads, self.q_head_dim)
        kv = self.kv_proj(hidden_last, lora_mapping=lora_mapping).squeeze(1)
        
        # Slicing once is faster than multiple match_dim calls
        q_nope = q[:, :, :self.qk_nope_head_dim]
        # DeepSeek-V2 MLA: k is also compressed/shared, here we take the first nope_head_dim
        k = kv[:, :self.qk_nope_head_dim].view(bsz, 1, self.qk_nope_head_dim)
        # v part starts after qk_nope_head_dim if it's packed this way in GGUF
        v_part = kv[:, self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim]
        if v_part.shape[-1] == 0: # Handle cases where V is same as K latent
            v = k
        else:
            v = v_part.view(bsz, 1, self.v_head_dim)
            if v.shape[-1] < self.v_head_dim:
                v = torch.nn.functional.pad(v, (0, self.v_head_dim - v.shape[-1]))

        from vllm.attention.ops.triton_paged_attn import triton_paged_attention
        output = triton_paged_attention(
            q_nope,
            k,
            v,
            kv_cache, attn_metadata["slot_mapping"], attn_metadata["seq_lens"], None, self.scale
        )
        return self.o_proj(output.view(bsz, 1, -1), lora_mapping=lora_mapping)

class DeepSeekV2MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.num_experts = config.n_routed_experts; self.topk = config.num_experts_per_tok
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_exps")
        self.w1_up = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up_exps")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_exps")
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False).cuda().half()
        self._expert_triplet_cache: "OrderedDict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._max_expert_cache_size = int(os.environ.get("FASTINFERENCE_DEEPSEEK_MOE_CACHE_SIZE", "128"))
        self._prewarm_budget = int(os.environ.get("FASTINFERENCE_DEEPSEEK_MOE_PREWARM", str(max(self.topk * 2, 16))))
        self._prewarm_done = False
        self._grouped_moe_enabled = os.environ.get("FASTINFERENCE_DEEPSEEK_GROUPED_MOE", "1") == "1"
        self._grouped_moe_fallback = os.environ.get("FASTINFERENCE_DEEPSEEK_GROUPED_MOE_FALLBACK", "1") == "1"
        self._grouped_moe_min_tokens = int(os.environ.get("FASTINFERENCE_DEEPSEEK_GROUPED_MOE_MIN_TOKENS", "2"))
        
        # Performance Tiers Control
        # Mode options: "full" (all cached), "dynamic" (LRU cache), "off" (real-time)
        self._cache_mode = os.environ.get("FASTINFERENCE_MOE_CACHE_MODE", "dynamic").lower()
        
        # FP8 Acceleration Control (Tier 0)
        self._fp8_enabled = os.environ.get("FASTINFERENCE_DEEPSEEK_FP8", "0") == "1"
        
        # Override legacy fused_moe flag if set
        if os.environ.get("FASTINFERENCE_DEEPSEEK_FUSED_MOE") == "1":
            self._cache_mode = "full"
        
        # LRU Cache Size (used in "dynamic" mode)
        self._max_expert_cache_size = int(os.environ.get("FASTINFERENCE_MOE_LRU_SIZE", "32"))
        
        self._fused_moe_enabled = (self._cache_mode == "full")
        self._all_experts_w1: Optional[torch.Tensor] = None
        self._all_experts_w1_up: Optional[torch.Tensor] = None
        self._all_experts_w2: Optional[torch.Tensor] = None
        
        # FP8 Specific Caches
        self._all_experts_w1_fp8: Optional[torch.Tensor] = None
        self._all_experts_w1_up_fp8: Optional[torch.Tensor] = None
        self._all_experts_w2_fp8: Optional[torch.Tensor] = None
        self._w1_scales: Optional[torch.Tensor] = None
        self._w1_up_scales: Optional[torch.Tensor] = None
        self._w2_scales: Optional[torch.Tensor] = None
        
        self._predequant_done = False
        
        if self._cache_mode == "full":
            tag = "FULL + FP8" if self._fp8_enabled else "FULL"
            print(f"[MoE Cache] Mode: {tag} (Pre-dequantizing all {self.num_experts} experts)")
        elif self._cache_mode == "dynamic":
            print(f"[MoE Cache] Mode: DYNAMIC (LRU size: {self._max_expert_cache_size} experts)")
        else:
            print(f"[MoE Cache] Mode: OFF")

    def _predequantize_all(self, dtype: torch.dtype):
        if self._predequant_done: return
        
        input_size = self.w1.input_size
        inter_size = self.w1.output_size // self.num_experts
        
        if getattr(self.w1, "gguf_shape", None) is not None:
            input_size = int(self.w1.gguf_shape[0])
            inter_size = int(self.w1.gguf_shape[1])

        print(f"[DeepSeek MoE] Pre-dequantizing all {self.num_experts} experts (inter={inter_size}, hidden={input_size}) to GPU ({dtype})...")
        
        with torch.no_grad():
            # Initial FP16 Caches
            self._all_experts_w1 = torch.empty((self.num_experts, inter_size, input_size), device="cuda", dtype=dtype)
            self._all_experts_w1_up = torch.empty((self.num_experts, inter_size, input_size), device="cuda", dtype=dtype)
            self._all_experts_w2 = torch.empty((self.num_experts, input_size, inter_size), device="cuda", dtype=dtype)
            
            for i in range(self.num_experts):
                w1, up, w2 = self._get_expert_triplet(i, dtype)
                self._all_experts_w1[i].copy_(w1.view(inter_size, input_size))
                self._all_experts_w1_up[i].copy_(up.view(inter_size, input_size))
                self._all_experts_w2[i].copy_(w2.view(input_size, inter_size))
                self._expert_triplet_cache.clear()
            
            # --- TIER 0: FP8 Conversion ---
            if self._fp8_enabled:
                print(f"[DeepSeek MoE] Converting experts to FP8 for 900+ TPS milestone...")
                f8_type = torch.float8_e4m3fn
                
                # Allocation for FP8 Weights
                self._all_experts_w1_fp8 = torch.empty_like(self._all_experts_w1, dtype=f8_type)
                self._all_experts_w1_up_fp8 = torch.empty_like(self._all_experts_w1_up, dtype=f8_type)
                self._all_experts_w2_fp8 = torch.empty_like(self._all_experts_w2, dtype=f8_type)
                
                # Allocation for Block Scales (assuming 64x64 blocks for our Triton kernel)
                # Scale per block: [Experts, Out//64, In//64]
                BS = 64
                self._w1_scales = torch.ones((self.num_experts, inter_size // BS, input_size // BS), device="cuda", dtype=torch.float32)
                self._w1_up_scales = torch.ones((self.num_experts, inter_size // BS, input_size // BS), device="cuda", dtype=torch.float32)
                self._w2_scales = torch.ones((self.num_experts, input_size // BS, inter_size // BS), device="cuda", dtype=torch.float32)
                
                def convert_to_fp8_with_scales(fp16_weights, fp8_out, scales):
                    for i in range(self.num_experts):
                        w = fp16_weights[i]
                        # Simplified scaling: max per block
                        # In real world, we'd use a kernel for this. For init, we do it via torch.
                        for r in range(scales.shape[1]):
                            for c in range(scales.shape[2]):
                                block = w[r*BS:(r+1)*BS, c*BS:(c+1)*BS]
                                s = block.abs().max().clamp(min=1e-12).item() / 448.0 # FP8 max is 448
                                scales[i, r, c] = s
                                fp8_out[i, r*BS:(r+1)*BS, c*BS:(c+1)*BS].copy_((block / s).to(f8_type))

                # Quick conversion (note: this is slow during init, but fast during inference)
                convert_to_fp8_with_scales(self._all_experts_w1, self._all_experts_w1_fp8, self._w1_scales)
                convert_to_fp8_with_scales(self._all_experts_w1_up, self._all_experts_w1_up_fp8, self._w1_up_scales)
                convert_to_fp8_with_scales(self._all_experts_w2, self._all_experts_w2_fp8, self._w2_scales)
                
                # Free FP16 if in full mode to save space
                if self._cache_mode == "full":
                    self._all_experts_w1 = None
                    self._all_experts_w1_up = None
                    self._all_experts_w2 = None

        self._predequant_done = True
        torch.cuda.empty_cache()
        print(f"[DeepSeek MoE] Pre-dequantization + FP8 Conversion complete.")

    def _dequantize_expert_matrix(self, layer: LiteLinear, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
        if getattr(layer, "qweight", None) is None:
            raise RuntimeError(f"DeepSeek expert tensor not loaded for expert={expert_idx} at {layer.prefix}")
        if getattr(layer, "gguf_quant_type", None) is None:
            raise RuntimeError(f"DeepSeek expert quant metadata missing for expert={expert_idx} at {layer.prefix}")
        
        # Determine dimension of experts
        if layer.qweight.dim() == 3:
            # GGUF 3D format: often [dim1, dim2, experts]
            if layer.qweight.shape[0] == self.num_experts: packed = layer.qweight[expert_idx].contiguous()
            elif layer.qweight.shape[2] == self.num_experts: packed = layer.qweight[:, :, expert_idx].contiguous()
            else: packed = layer.qweight[expert_idx].contiguous() # Fallback
        else:
            packed = layer.qweight[expert_idx].contiguous()

        # Get m, n from weight properties
        # For Experts, GGUF stores weights differently
        if layer.gguf_shape is not None:
            # Based on testing, ggml_dequantize_fallback returns [n, m]
            # where m is the first arg, n is the second.
            # We want [out, in], so m=out_dim, n=in_dim
            # For w1: out=inter, in=hidden. For GLM: inter=1536, hidden=2048
            # gguf_shape was set to (2048, 1536) in model_loader.
            in_dim, out_dim = int(layer.gguf_shape[0]), int(layer.gguf_shape[1])
            m_dim, n_dim = out_dim, in_dim 
        else:
            m_dim = layer.output_size // self.num_experts
            n_dim = layer.input_size

        return ggml_dequantize_fallback(
            packed,
            int(layer.gguf_quant_type),
            m_dim,
            n_dim,
            dtype,
        )

    def _get_expert_triplet(self, expert_idx: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache_mode == "off":
            gate_w = self._dequantize_expert_matrix(self.w1, expert_idx, dtype)
            up_w = self._dequantize_expert_matrix(self.w1_up, expert_idx, dtype)
            down_w = self._dequantize_expert_matrix(self.w2, expert_idx, dtype)
            return (gate_w, up_w, down_w)

        cache_key = (expert_idx, dtype)
        cached = self._expert_triplet_cache.get(cache_key)
        if cached is not None:
            self._expert_triplet_cache.move_to_end(cache_key)
            return cached
        
        gate_w = self._dequantize_expert_matrix(self.w1, expert_idx, dtype)
        up_w = self._dequantize_expert_matrix(self.w1_up, expert_idx, dtype)
        down_w = self._dequantize_expert_matrix(self.w2, expert_idx, dtype)
        triplet = (gate_w, up_w, down_w)
        
        self._expert_triplet_cache[cache_key] = triplet
        if len(self._expert_triplet_cache) > self._max_expert_cache_size:
            self._expert_triplet_cache.popitem(last=False)
        return triplet

    def _prewarm_hot_experts(self, router_logits: torch.Tensor, dtype: torch.dtype) -> None:
        if self._prewarm_done:
            return
        prewarm_k = min(self.num_experts, max(self.topk, self._prewarm_budget))
        mean_router = router_logits.float().mean(dim=0)
        _, warm_ids = torch.topk(mean_router, k=prewarm_k, dim=-1)
        for expert_id in warm_ids.tolist():
            self._get_expert_triplet(int(expert_id), dtype)
        self._prewarm_done = True

    def _forward_token_loop(self, curr_x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(curr_x)
        for token_idx in range(curr_x.shape[0]):
            hidden_vec = curr_x[token_idx]
            token_out = torch.zeros_like(hidden_vec)
            for route_idx, expert_id in enumerate(topk_ids[token_idx]):
                gate_w, up_w, down_w = self._get_expert_triplet(int(expert_id.item()), curr_x.dtype)
                gate_act = torch.mv(gate_w, hidden_vec)
                up_act = torch.mv(up_w, hidden_vec)
                mixed = torch.nn.functional.silu(gate_act) * up_act
                expert_out = torch.mv(down_w, mixed)
                token_out = token_out + expert_out * topk_weights[token_idx, route_idx]
            output[token_idx] = token_out
        return output

    def _forward_grouped(self, curr_x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        num_tokens = curr_x.shape[0]
        flat_expert_ids = topk_ids.reshape(-1)
        flat_token_ids = torch.arange(num_tokens, device=curr_x.device, dtype=torch.long).repeat_interleave(self.topk)
        flat_route_weights = topk_weights.reshape(-1).to(curr_x.dtype)
        
        sorted_ids, sort_idx = torch.sort(flat_expert_ids)
        sorted_token_ids = flat_token_ids[sort_idx]
        sorted_weights = flat_route_weights[sort_idx]
        
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        output = torch.zeros_like(curr_x)

        # --- Tier 0: FP8 Input Preparation ---
        if self._fp8_enabled and self._predequant_done:
            from vllm.kernels.triton.fp8_gemm import fp8_block_gemm
            # Convert inputs to FP8 once
            # Note: For real max-perf, we need per-token scaling for inputs
            # For this milestone, we use a global scale for Warmup
            x_f8 = curr_x.to(torch.float8_e4m3fn)
            x_scale = torch.ones((num_tokens, curr_x.shape[1] // 64), device="cuda", dtype=torch.float32) * (curr_x.abs().max() / 448.0)
        
        curr_off = 0
        for i in range(len(unique_ids)):
            expert_id = int(unique_ids[i].item())
            count = int(counts[i].item())
            
            token_indices = sorted_token_ids[curr_off : curr_off + count]
            route_weights = sorted_weights[curr_off : curr_off + count].unsqueeze(-1)
            curr_off += count
            
            hidden_batch = curr_x.index_select(0, token_indices)
            
            # --- FP8 Optimized Path ---
            if self._fp8_enabled and self._predequant_done and self._all_experts_w1_fp8 is not None:
                batch_f8 = x_f8.index_select(0, token_indices)
                batch_scales = x_scale.index_select(0, token_indices)
                
                # W1/W1_Up: [count, inter]
                # We reuse our fp8_block_gemm kernel
                gate_act = fp8_block_gemm(batch_f8, self._all_experts_w1_fp8[expert_id].T, batch_scales, self._w1_scales[expert_id])
                up_act = fp8_block_gemm(batch_f8, self._all_experts_w1_up_fp8[expert_id].T, batch_scales, self._w1_up_scales[expert_id])
                
                mixed = torch.nn.functional.silu(gate_act) * up_act
                
                # W2: [count, hidden]
                mixed_f8 = mixed.to(torch.float8_e4m3fn)
                mixed_scale = torch.ones((count, mixed.shape[1] // 64), device="cuda", dtype=torch.float32) * (mixed.abs().max() / 448.0)
                expert_out = fp8_block_gemm(mixed_f8, self._all_experts_w2_fp8[expert_id].T, mixed_scale, self._w2_scales[expert_id])
            else:
                # Standard FP16 Path
                if self._predequant_done and self._all_experts_w1 is not None:
                    gate_w, up_w, down_w = self._all_experts_w1[expert_id], self._all_experts_w1_up[expert_id], self._all_experts_w2[expert_id]
                else:
                    gate_w, up_w, down_w = self._get_expert_triplet(expert_id, curr_x.dtype)
                
                gate_act = hidden_batch @ gate_w.transpose(0, 1)
                up_act = hidden_batch @ up_w.transpose(0, 1)
                mixed = torch.nn.functional.silu(gate_act) * up_act
                expert_out = mixed @ down_w.transpose(0, 1)
            
            output.index_add_(0, token_indices, expert_out.to(output.dtype) * route_weights)
        return output

    def forward(self, x, lora_mapping=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        curr_x = x[:, -1:, :].view(-1, x.shape[-1])
        router_logits = self.gate(curr_x.to(self.gate.weight.dtype)).to(curr_x.dtype)
        
        if not self._prewarm_done:
            self._prewarm_hot_experts(router_logits, curr_x.dtype)
            
        topk_vals, topk_ids = torch.topk(router_logits, k=self.topk, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        
        # Performance Tier 1: Fused MoE with Full GPU Expert Cache
        # Enabled for BS > 1 and when VRAM allows.
        if self._fused_moe_enabled and curr_x.shape[0] >= self._grouped_moe_min_tokens:
            if not self._predequant_done:
                self._predequantize_all(curr_x.dtype)
            
            # Using torch.ops for grouped GEMM on pre-dequantized weights is very fast.
            # We use the optimized Tiered MoE path in our fused_moe dispatcher.
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
            # DeepSeek-V2 MoE has separate gate (w1) and up (w1_up).
            # We cat them for the fused_moe call to match its standard interface.
            # However, for max TPS, we perform them separately.
            
            # Tier 1 optimization: 
            # 1. Dispatch tokens to unique experts
            # 2. Batched GEMM using @ for all experts
            # 3. Fuse SiLU and Add
            output = self._forward_grouped(curr_x, topk_ids, topk_weights)
            return output.view(x.shape[0], 1, -1)
            
        use_grouped = self._grouped_moe_enabled and curr_x.shape[0] >= self._grouped_moe_min_tokens
        if use_grouped:
            try:
                output = self._forward_grouped(curr_x, topk_ids, topk_weights)
            except RuntimeError:
                if not self._grouped_moe_fallback:
                    raise
                output = self._forward_token_loop(curr_x, topk_ids, topk_weights)
        else:
            output = self._forward_token_loop(curr_x, topk_ids, topk_weights)
        return output.view(x.shape[0], 1, -1)

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
        self.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate")
        self.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        g = self.gate_proj(x[:, -1:, :], lora_mapping=lora_mapping)
        u = self.up_proj(x[:, -1:, :], lora_mapping=lora_mapping)
        return self.down_proj(self.act(g) * u, lora_mapping=lora_mapping)

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
