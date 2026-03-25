# SPDX-License-Identifier: Apache-2.0
import os
from math import prod
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
import numpy as np
import gguf
import json
import re
import gc
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_awq_pytorch,
    dequantize_symmetric_packed_int4_pytorch,
    GGUFWeight,
)
from vllm.model_executor.moe_fp8_utils import (
    dims_ok_for_moe_fp8,
    fp8_block_quantize_2d,
    moe_fp8_block_size,
    qwen35_moe_fp8_enabled,
    qwen35_moe_offload_enabled,
)
from vllm.model_executor.moe_gguf_packed import (
    gguf_quant_type_supported_for_moe_packed,
    numpy_gguf_data_to_packed_2d,
)

# --- GGUF load peak RSS (why "temporary buffers" matter) ---
# For each quantized tensor, gguf.dequantize() builds a full float32 numpy array (dequant_np),
# then we materialize a torch tensor (often FP16). Peak RSS is dominated by:
#   (1) compressed tensor bytes still reachable from GGUFReader.tensor_map,
#   (2) dequant_np (full logical size, float32),
#   (3) torch tensor after from_numpy / dtype cast (may duplicate if cast copies),
#   (4) MoE: full [E, I, H] gate and up tensors before copy into Parameters (sequential ge vs ue halves (2)).
# Mitigations: del ge/ue as soon as possible, gc.collect() per layer, avoid extra cats/temps in FP8 path.
# Optional FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=1: keep ffn_*_exps as uint8 packed rows (no full-blob dequant at load).


def _dequantize_gguf_tensor(gguf_tensor: Any, device: str, dtype: torch.dtype, target_shape: Optional[torch.Size] = None) -> Optional[torch.Tensor]:
    q_type = int(gguf_tensor.tensor_type)
    # F32 (0) and F16 (1): row-major bytes match GGUF metadata shape; do NOT view(target_shape)
    # first — that scrambles layout when GGUF order is the transpose of PyTorch (e.g. embed, Linear).
    if q_type in [0, 1]:
        # Prefer asarray + from_numpy over always copy=True (reduces peak RSS when buffer is writable).
        w_np = np.asarray(gguf_tensor.data)
        if not getattr(w_np, "flags", None) or not w_np.flags.writeable:
            w_np = w_np.copy()
        raw = torch.from_numpy(w_np).to(device=device)
        tname = str(getattr(gguf_tensor, "name", "") or "")
        # Depthwise conv1d: GGUF dims may not match PyTorch Conv1d layout; contiguous bytes match HF
        # when viewed as target_shape [C, 1, K] (see scripts/qwen35_gguf_alignment_audit.py).
        if "ssm_conv1d.weight" in tname and target_shape is not None and len(target_shape) == 3:
            if raw.numel() == prod(target_shape):
                try:
                    return raw.view(*target_shape).to(dtype=dtype)
                except Exception:
                    pass
        gs = tuple(int(x) for x in gguf_tensor.shape)
        expected = prod(gs) if gs else 0
        if gs and raw.numel() == expected:
            res = raw.view(*gs)
        else:
            res = raw
        res = res.to(dtype=dtype)
        if res.ndim == 2 and target_shape is not None and len(target_shape) == 2:
            tr, tc = target_shape[0], target_shape[1]
            if res.shape == (tr, tc):
                return res
            if res.shape == (tc, tr):
                return res.T.contiguous()
        if target_shape is not None:
            try:
                return res.view(target_shape)
            except Exception:
                return res
        return res
    
    if target_shape is not None:
        if len(target_shape) == 1: R, C = 1, target_shape[0]
        elif len(target_shape) == 2: R, C = target_shape[0], target_shape[1]
        else:
            # 3D: [Experts, Rows, Cols] -> Logical R = Experts*Rows, C = Cols
            C = target_shape[-1]
            R = 1
            for dim in target_shape[:-1]: R *= dim
    else:
        # Physical fallback from GGUF shape (note: GGUF shape is reversed)
        ps = gguf_tensor.shape
        C = ps[0]
        R = 1
        for dim in ps[1:]: R *= dim

    # Universal dequant via gguf library — handles Q4_K, Q6_K, Q5_K, Q8_0, IQ, BF16, etc.
    try:
        from gguf import dequantize as gguf_dequantize, GGMLQuantizationType
        w_np = np.asarray(gguf_tensor.data)
        if not getattr(w_np, "flags", None) or not w_np.flags.writeable:
            w_np = w_np.copy()
        dequant_np = gguf_dequantize(w_np, GGMLQuantizationType(q_type))
        del w_np
        # One numpy->torch bridge; avoid redundant np.array(..., copy=True) on dequant output.
        res = torch.from_numpy(dequant_np).to(device=device, dtype=dtype)
        # Note: do not del dequant_np here — from_numpy may share storage with float32 CPU tensors.
        # gguf.dequantize returns row-major logical weights, often already [out, in] for Linear
        # (e.g. token_embd (V,H)) even when GGUF tensor metadata lists the transpose (H,V).
        # Reshaping to metadata dims would permute elements and destroy alignment vs HF.
        if target_shape is not None and len(target_shape) == 2:
            tr, tc = int(target_shape[0]), int(target_shape[1])
            if res.ndim == 2 and res.numel() == tr * tc:
                if res.shape == (tr, tc):
                    return res
                if res.shape == (tc, tr):
                    return res.T.contiguous()
        # Reshape using GGUF metadata when layout is still ambiguous (e.g. flattened).
        gs = tuple(int(x) for x in gguf_tensor.shape)
        if len(gs) == 2 and res.numel() == gs[0] * gs[1]:
            res = res.reshape(gs[0], gs[1])
        elif len(gs) == 1 and res.numel() == gs[0]:
            res = res.reshape(gs[0])
        else:
            total = R * C
            if res.numel() >= total:
                res = res.view(-1)[:total].view(R, C)
            else:
                out = torch.zeros(total, device=res.device, dtype=dtype)
                out[:res.numel()] = res.view(-1)
                res = out.view(R, C)
        # Map GGUF 2D layout -> PyTorch [out, in] when target_shape is the transpose.
        if target_shape is not None and res.ndim == 2 and len(target_shape) == 2:
            tr, tc = target_shape[0], target_shape[1]
            if res.shape == (tr, tc):
                return res
            if res.shape == (tc, tr):
                return res.T.contiguous()
        if target_shape is not None:
            try:
                return res.view(target_shape)
            except Exception:
                return res
        return res
    except Exception as e:
        print(f"    [Warning] gguf dequant failed for type={q_type}: {e}")
        return None

def _reorder_v_heads_ggml_to_hf(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_per_k: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Convert GGML tiled V-head layout back to Hugging Face grouped layout.

    llama.cpp export (_LinearAttentionVReorderBase) maps grouped -> tiled via
    reshape(..., num_k_heads, num_v_per_k, head_dim) then swap dims (inverse uses
    reshape(..., num_v_per_k, num_k_heads, head_dim) then swap — not the same as forward).
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    t = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def _qwen35_linear_attn_gguf_to_hf(
    source: torch.Tensor,
    gguf_tensor_name: str,
    hf_config: Any,
) -> torch.Tensor:
    """
    Undo GGML V-head tiling for Qwen3.5 linear-attn tensors (see llama.cpp
    Qwen3_5TextModel / _LinearAttentionVReorderBase).
    """
    mt = getattr(hf_config, "model_type", "") or ""
    if mt not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"):
        return source
    num_k = int(getattr(hf_config, "linear_num_key_heads", 0) or 0)
    num_v = int(getattr(hf_config, "linear_num_value_heads", 0) or 0)
    if num_k <= 0 or num_v <= 0 or num_k == num_v:
        return source
    head_k_dim = int(getattr(hf_config, "linear_key_head_dim", 0) or 0)
    head_v_dim = int(getattr(hf_config, "linear_value_head_dim", 0) or 0)
    if head_k_dim <= 0 or head_v_dim <= 0:
        return source
    num_v_per_k = num_v // num_k
    if num_v_per_k * num_k != num_v:
        return source

    q_dim = head_k_dim * num_k
    k_dim = head_k_dim * num_k
    v_dim = head_v_dim * num_v

    if "attn_qkv.weight" in gguf_tensor_name:
        if source.shape[0] != q_dim + k_dim + v_dim:
            return source
        q = source[:q_dim]
        k = source[q_dim : q_dim + k_dim]
        v = source[q_dim + k_dim :]
        v = _reorder_v_heads_ggml_to_hf(v, 0, num_k, num_v_per_k, head_v_dim)
        return torch.cat([q, k, v], dim=0)
    if "attn_gate.weight" in gguf_tensor_name:
        if source.shape[0] != v_dim:
            return source
        return _reorder_v_heads_ggml_to_hf(source, 0, num_k, num_v_per_k, head_v_dim)
    if "ssm_out.weight" in gguf_tensor_name:
        return _reorder_v_heads_ggml_to_hf(source, 1, num_k, num_v_per_k, head_v_dim)
    if "ssm_alpha.weight" in gguf_tensor_name or "ssm_beta.weight" in gguf_tensor_name:
        if source.shape[0] != num_v:
            return source
        return _reorder_v_heads_ggml_to_hf(source, 0, num_k, num_v_per_k, 1)
    return source


def _qwen35_rms_norm_weight_gguf_to_hf(
    source: torch.Tensor,
    gguf_tensor_name: str,
    hf_config: Any,
) -> torch.Tensor:
    """
    llama.cpp Qwen3Next/Qwen3.5 conversion adds +1 to RMSNorm .weight for GGML, except
    GatedDeltaNet linear_attn.norm (blk.*.ssm_norm.weight). Hugging Face checkpoints store
    the raw scale without that offset.
    """
    mt = getattr(hf_config, "model_type", "") or ""
    if mt not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"):
        return source
    if "norm.weight" not in gguf_tensor_name:
        return source
    # HF linear_attn gated norm is not shifted in convert_hf_to_gguf
    if "ssm_norm.weight" in gguf_tensor_name:
        return source
    if "linear_attn.norm" in gguf_tensor_name:
        return source
    if source.ndim == 1:
        return source - 1
    return source


def _qwen35_linear_attn_1d_gguf_to_hf(
    source: torch.Tensor,
    hf_config: Any,
) -> torch.Tensor:
    """A_log / dt_bias: reorder num_v_heads elements from tiled to grouped."""
    mt = getattr(hf_config, "model_type", "") or ""
    if mt not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"):
        return source
    num_k = int(getattr(hf_config, "linear_num_key_heads", 0) or 0)
    num_v = int(getattr(hf_config, "linear_num_value_heads", 0) or 0)
    if num_k <= 0 or num_v <= 0 or num_k == num_v:
        return source
    num_v_per_k = num_v // num_k
    if source.ndim != 1 or source.shape[0] != num_v:
        return source
    return _reorder_v_heads_ggml_to_hf(source.unsqueeze(-1), 0, num_k, num_v_per_k, 1).squeeze(-1)


def _qwen35_conv1d_channels_gguf_to_hf(
    conv_weight: torch.Tensor,
    hf_config: Any,
) -> torch.Tensor:
    """
    Reorder V channel block in depthwise conv1d [C, 1, K] along dim 0 (see llama.cpp).
    """
    mt = getattr(hf_config, "model_type", "") or ""
    if mt not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"):
        return conv_weight
    num_k = int(getattr(hf_config, "linear_num_key_heads", 0) or 0)
    num_v = int(getattr(hf_config, "linear_num_value_heads", 0) or 0)
    head_k_dim = int(getattr(hf_config, "linear_key_head_dim", 0) or 0)
    head_v_dim = int(getattr(hf_config, "linear_value_head_dim", 0) or 0)
    if num_k <= 0 or num_v <= 0 or num_k == num_v or head_k_dim <= 0 or head_v_dim <= 0:
        return conv_weight
    num_v_per_k = num_v // num_k
    qk_channels = head_k_dim * num_k * 2
    if conv_weight.ndim != 3 or conv_weight.shape[0] < qk_channels + 1:
        return conv_weight
    data = conv_weight.squeeze(1)
    qk_part = data[:qk_channels]
    v_part = data[qk_channels:]
    v_part = _reorder_v_heads_ggml_to_hf(v_part, 0, num_k, num_v_per_k, head_v_dim)
    out = torch.cat([qk_part, v_part], dim=0).unsqueeze(1)
    return out.contiguous()


def _try_load_legacy_flat_linear_attn_norm_from_safetensors(
    p_name: str,
    p: torch.Tensor,
    sd_keys: List[str],
    sd: Dict[str, torch.Tensor],
) -> bool:
    """
    Old Lite checkpoints stored gated linear-attn RMS scale as a flat Parameter:
      model.layers.N.linear_attn.norm
    New module uses Qwen3_5RMSNormGated with:
      model.layers.N.linear_attn.norm.weight
    Accept legacy keys when shapes match (same layer index).
    """
    if not p_name.endswith(".linear_attn.norm.weight"):
        return False
    layer_m = re.search(r"layers\.(\d+)\.", p_name)
    if not layer_m:
        return False
    layer_idx = layer_m.group(1)
    for k in sd_keys:
        if f"layers.{layer_idx}." not in k:
            continue
        if k.endswith(".linear_attn.norm.weight"):
            continue
        if not k.endswith(".linear_attn.norm"):
            continue
        tensor = sd[k]
        if tensor.shape != p.shape:
            continue
        p.data.copy_(tensor.to(dtype=p.dtype))
        return True
    return False


def _load_param_from_gguf_tensor(
    param: Any,
    gguf_tensor: Any,
    slice_offset: int = 0,
    hf_config: Optional[Any] = None,
) -> None:
    actual_param = param; logical_shape = None
    if isinstance(param, LiteLinear): logical_shape = torch.Size([param.output_size, param.input_size]); actual_param = param.weight
    elif isinstance(param, nn.Module) and hasattr(param, "weight"): actual_param = param.weight
    if not isinstance(actual_param, torch.Tensor) and logical_shape is None: return
    target_shape = logical_shape if logical_shape is not None else actual_param.shape
    # Use universal dequant via gguf library
    source = _dequantize_gguf_tensor(gguf_tensor, "cpu", torch.float16, target_shape)
    if source is None: return

    tname = str(getattr(gguf_tensor, "name", "") or "")
    if hf_config is not None and any(
        x in tname for x in ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight", "ssm_beta.weight")
    ):
        source = _qwen35_linear_attn_gguf_to_hf(source, tname, hf_config)
    if hf_config is not None:
        source = _qwen35_rms_norm_weight_gguf_to_hf(source, tname, hf_config)

    # Special case for ssm_conv1d.weight which is often [kernel, channels] in GGUF
    # but [channels, 1, kernel] in PyTorch nn.Conv1d(groups=channels)
    if "ssm_conv1d.weight" in gguf_tensor.name:
        if source.ndim == 2 and source.shape[0] < source.shape[1]:
            # [4, 8192] -> [8192, 1, 4]
            source = source.T.unsqueeze(1)
        elif source.ndim == 1:
            # Flattened?
            pass

    if isinstance(param, LiteLinear):
        # Module-level load: can replace the Parameter
        if param.weight.numel() == 0 or param.weight.shape != source.shape:
            param.weight = nn.Parameter(source.to(dtype=torch.float16), requires_grad=False)
        else:
            param.weight.data.copy_(source.to(dtype=param.weight.dtype))
    if isinstance(actual_param, nn.Parameter):
        # Parameter update
        if actual_param.numel() == 0:
            # Resize empty parameter
            actual_param.data = source.to(dtype=actual_param.dtype)
            print(f"    [Verify] {gguf_tensor.name} resized to {actual_param.shape}, numel={actual_param.numel()}")
        elif actual_param.shape == source.shape:
            actual_param.data.copy_(source)
        else:
            # Heuristic mapping (legacy); prefer correct layout from _dequantize_gguf_tensor.
            try:
                o, i = actual_param.shape[0], actual_param.shape[1]
                if source.ndim == 2 and source.shape == (i, o):
                    actual_param.data.copy_(source.T.contiguous())
                elif source.shape[0] == actual_param.shape[1] and source.shape[1] >= actual_param.shape[0]:
                    actual_param.data.copy_(source[:, :actual_param.shape[0]].T)
                elif source.shape[1] == actual_param.shape[1] and source.shape[0] >= actual_param.shape[0]:
                    actual_param.data.copy_(source[:actual_param.shape[0], :])
            except Exception:
                pass
    elif isinstance(actual_param, torch.Tensor):
        if actual_param.shape == source.shape:
            actual_param.data.copy_(source)
    try:
        del source
    except NameError:
        pass

def _load_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    print(f">>> Mapping GGUF weights from {gguf_file}...")
    reader = gguf.GGUFReader(gguf_file); tensor_map = {t.name: t for t in reader.tensors}
    # Per-layer only (global tensors loaded once below — avoids N× redundant embed/lm_head writes).
    map_rules = {
        "blk.{i}.attn_norm.weight": ["model.layers.{i}.input_layernorm.weight"],
        "blk.{i}.ffn_norm.weight": ["model.layers.{i}.post_attention_layernorm.weight"],
        # Qwen3.5 GGUF uses 'post_attention_norm' instead of 'ffn_norm'
        "blk.{i}.post_attention_norm.weight": ["model.layers.{i}.post_attention_layernorm.weight"],
        # DeepSeek V2 MLA layers
        "blk.{i}.attn_q_a.weight": ["model.layers.{i}.self_attn.q_a_proj.weight"],
        "blk.{i}.attn_q_a_norm.weight": ["model.layers.{i}.self_attn.q_a_layernorm.weight"],
        "blk.{i}.attn_q_b.weight": ["model.layers.{i}.self_attn.q_b_proj.weight"],
        "blk.{i}.attn_kv_a_mqa.weight": ["model.layers.{i}.self_attn.kv_a_proj.weight"],
        "blk.{i}.attn_kv_a_norm.weight": ["model.layers.{i}.self_attn.kv_a_layernorm.weight"],
        "blk.{i}.attn_output.weight": ["model.layers.{i}.self_attn.o_proj.weight"],
        # Qwen3.5 full-attention layers (standard GGUF names)
        "blk.{i}.attn_q.weight": ["model.layers.{i}.self_attn.q_proj.weight"],
        "blk.{i}.attn_k.weight": ["model.layers.{i}.self_attn.k_proj.weight"],
        "blk.{i}.attn_v.weight": ["model.layers.{i}.self_attn.v_proj.weight"],
        "blk.{i}.attn_q_norm.weight": ["model.layers.{i}.self_attn.q_norm.weight"],
        "blk.{i}.attn_k_norm.weight": ["model.layers.{i}.self_attn.k_norm.weight"],
        # Qwen3.5 linear-attention / GatedDeltaNet (SSM-style GGUF names)
        "blk.{i}.attn_qkv.weight": ["model.layers.{i}.linear_attn.in_proj_qkv.weight"],
        "blk.{i}.attn_gate.weight": ["model.layers.{i}.linear_attn.in_proj_z.weight"],
        "blk.{i}.ssm_norm.weight": ["model.layers.{i}.linear_attn.norm.weight"],
        "blk.{i}.ssm_out.weight": ["model.layers.{i}.linear_attn.out_proj.weight"],
        "blk.{i}.ssm_alpha.weight": ["model.layers.{i}.linear_attn.in_proj_a.weight"],
        "blk.{i}.ssm_beta.weight": ["model.layers.{i}.linear_attn.in_proj_b.weight"],
        # Qwen3.5 legacy HF-style GGUF export
        "blk.{i}.linear_attn.norm.weight": ["model.layers.{i}.linear_attn.norm.weight"],
        # Qwen3.5 standard MLP (dense non-MoE layers)
        "blk.{i}.ffn_gate.weight": ["model.layers.{i}.mlp.gate_proj.weight"],
        "blk.{i}.ffn_up.weight": ["model.layers.{i}.mlp.up_proj.weight"],
        "blk.{i}.ffn_down.weight": ["model.layers.{i}.mlp.down_proj.weight"],
        # Qwen3.5 MoE (HF: mlp.gate / mlp.experts / mlp.shared_expert / mlp.shared_expert_gate)
        "blk.{i}.ffn_gate_inp.weight": ["model.layers.{i}.mlp.gate.weight"],
        "blk.{i}.ffn_gate_shexp.weight": ["model.layers.{i}.mlp.shared_expert.gate_proj.weight"],
        "blk.{i}.ffn_up_shexp.weight": ["model.layers.{i}.mlp.shared_expert.up_proj.weight"],
        "blk.{i}.ffn_down_shexp.weight": ["model.layers.{i}.mlp.shared_expert.down_proj.weight"],
        # llama.cpp may name this ffn_gate_sw_shexp or ffn_gate_inp_shexp (same HF tensor)
        "blk.{i}.ffn_gate_sw_shexp.weight": ["model.layers.{i}.mlp.shared_expert_gate.weight"],
        "blk.{i}.ffn_gate_inp_shexp.weight": ["model.layers.{i}.mlp.shared_expert_gate.weight"],
    }
    modules = dict(model.named_modules()); params = dict(model.named_parameters())

    # Global tensors (not tied to blk.{i}): load once.
    global_map = {
        "token_embd.weight": ["model.embed_tokens.weight"],
        "output_norm.weight": ["model.norm.weight"],
        "output.weight": ["lm_head.weight"],
    }
    loaded_count = 0
    for g_key, m_paths in global_map.items():
        if g_key not in tensor_map:
            continue
        for m_path in m_paths:
            m_base = m_path.rsplit(".", 1)[0] if "." in m_path else m_path
            if m_base in modules:
                print(f"    [Load Module] {g_key} -> {m_base}")
                _load_param_from_gguf_tensor(modules[m_base], tensor_map[g_key], hf_config=hf_config)
                loaded_count += 1
                break
            if m_path in params:
                print(f"    [Load Param] {g_key} -> {m_path}")
                _load_param_from_gguf_tensor(params[m_path], tensor_map[g_key], hf_config=hf_config)
                loaded_count += 1
                break

    # Metadata for MoE shapes — support both DeepSeek and Qwen naming
    num_experts = getattr(hf_config, "num_experts",
                  getattr(hf_config, "n_routed_experts", 64))
    moe_inter = getattr(hf_config, "moe_intermediate_size", 1536)
    hidden = getattr(hf_config, "hidden_size", 2048)
    
    num_layers = getattr(hf_config, "num_hidden_layers", 28)
    for i in range(num_layers):
        for g_pat, m_paths in map_rules.items():
            g_key = g_pat.format(i=i)
            if g_key in tensor_map:
                for m_path in m_paths:
                    target_path = m_path.format(i=i)
                    m_base = target_path.rsplit(".", 1)[0] if "." in target_path else target_path
                    if m_base in modules:
                        print(f"    [Load Module] {g_key} -> {m_base}")
                        _load_param_from_gguf_tensor(modules[m_base], tensor_map[g_key], hf_config=hf_config)
                        loaded_count += 1
                        break
                    elif target_path in params:
                        print(f"    [Load Param] {g_key} -> {target_path}")
                        _load_param_from_gguf_tensor(params[target_path], tensor_map[g_key], hf_config=hf_config)
                        loaded_count += 1
                        break

        # --- Qwen3.5 special tensors (non-LiteLinear: scalars, biases, conv1d) ---

        # ssm_a -> linear_attn.A_log (1D float32, not quantized)
        ssm_a_key = f"blk.{i}.ssm_a"
        a_log_path = f"model.layers.{i}.linear_attn.A_log"
        if ssm_a_key in tensor_map and a_log_path in params:
            src = _dequantize_gguf_tensor(tensor_map[ssm_a_key], "cpu", torch.float32,
                                          target_shape=params[a_log_path].shape)
            if src is not None:
                src = _qwen35_linear_attn_1d_gguf_to_hf(src, hf_config)
                params[a_log_path].data.copy_(src.to(dtype=params[a_log_path].dtype))
                loaded_count += 1

        # ssm_dt.bias -> linear_attn.dt_bias (1D float32, not quantized)
        dt_bias_key = f"blk.{i}.ssm_dt.bias"
        dt_bias_path = f"model.layers.{i}.linear_attn.dt_bias"
        if dt_bias_key in tensor_map and dt_bias_path in params:
            src = _dequantize_gguf_tensor(tensor_map[dt_bias_key], "cpu", torch.float32,
                                          target_shape=params[dt_bias_path].shape)
            if src is not None:
                src = _qwen35_linear_attn_1d_gguf_to_hf(src, hf_config)
                params[dt_bias_path].data.copy_(src.to(dtype=params[dt_bias_path].dtype))
                loaded_count += 1

        # ssm_conv1d.weight -> linear_attn.conv1d.weight
        # GGUF stores conv1d as F32, shape metadata is reversed.
        # PyTorch Conv1d(groups=channels) expects [channels, 1, kernel_size]
        conv_key = f"blk.{i}.ssm_conv1d.weight"
        conv_path = f"model.layers.{i}.linear_attn.conv1d.weight"
        if conv_key in tensor_map and conv_path in params:
            tgt_param = params[conv_path]
            gguf_t = tensor_map[conv_key]
            try:
                # F32 conv: use same path as audit — raw.view(metadata) can disagree with HF;
                # view to target [C,1,K] matches safetensors byte order.
                src_conv = _dequantize_gguf_tensor(
                    gguf_t,
                    "cpu",
                    tgt_param.dtype,
                    target_shape=tgt_param.shape,
                )
                if src_conv is not None and src_conv.shape == tgt_param.shape:
                    src_conv = _qwen35_conv1d_channels_gguf_to_hf(src_conv, hf_config)
                    tgt_param.data.copy_(src_conv.to(dtype=tgt_param.dtype))
                    loaded_count += 1
                elif src_conv is None:
                    print(f"    [Warning] Conv1d dequant failed layer {i}")
            except Exception as e:
                print(f"    [Warning] Conv1d load failed layer {i}: {e}")

        # Legacy GGUF: flat tensor name blk.N.linear_attn.norm -> norm.weight module
        legacy_lin_norm = f"blk.{i}.linear_attn.norm"
        target_lin_norm_w = f"model.layers.{i}.linear_attn.norm.weight"
        if legacy_lin_norm in tensor_map and target_lin_norm_w in params:
            tgt = params[target_lin_norm_w]
            src = _dequantize_gguf_tensor(tensor_map[legacy_lin_norm], "cpu", torch.float16, target_shape=tgt.shape)
            if src is not None and src.shape == tgt.shape:
                tgt.data.copy_(src.to(dtype=tgt.dtype))
                loaded_count += 1

        kb_key = f"blk.{i}.attn_k_b.weight"; vb_key = f"blk.{i}.attn_v_b.weight"
        if kb_key in tensor_map and vb_key in tensor_map:
            kb = _dequantize_gguf_tensor(tensor_map[kb_key], "cpu", torch.float16)
            vb = _dequantize_gguf_tensor(tensor_map[vb_key], "cpu", torch.float16)
            if kb is not None and vb is not None:
                fused_kv_b = torch.cat([kb, vb], dim=0)
                m_path = f"model.layers.{i}.self_attn.kv_b_proj"
                if m_path in modules: modules[m_path].weight = nn.Parameter(fused_kv_b, requires_grad=False); loaded_count += 1
        
        # MoE routed experts: fuse gate+up -> gate_up_proj (HF), load down_proj
        ge_key = f"blk.{i}.ffn_gate_exps.weight"
        ue_key = f"blk.{i}.ffn_up_exps.weight"
        de_key = f"blk.{i}.ffn_down_exps.weight"
        m_moe_path = f"model.layers.{i}.mlp"
        if ge_key in tensor_map and ue_key in tensor_map and m_moe_path in modules:
            m_moe = modules[m_moe_path]
            exp_mod = getattr(m_moe, "experts", None)
            if exp_mod is None:
                pass
            elif getattr(exp_mod, "moe_gguf_packed", False):
                if de_key not in tensor_map:
                    raise RuntimeError(
                        f"MoE packed GGUF requires {de_key} in the GGUF file (layer {i})."
                    )
                t_shape_g = torch.Size([num_experts, moe_inter, hidden])
                t_shape_d = torch.Size([num_experts, hidden, moe_inter])
                ge_t = tensor_map[ge_key]
                ue_t = tensor_map[ue_key]
                qg = int(ge_t.tensor_type)
                qu = int(ue_t.tensor_type)
                if not gguf_quant_type_supported_for_moe_packed(qg) or not gguf_quant_type_supported_for_moe_packed(
                    qu
                ):
                    raise RuntimeError(
                        f"MoE packed GGUF requires quant types Q4_0/Q4_K/Q6_K; got gate={qg} up={qu}"
                    )
                gate_np = numpy_gguf_data_to_packed_2d(np.asarray(ge_t.data), tuple(t_shape_g), qg)
                up_np = numpy_gguf_data_to_packed_2d(np.asarray(ue_t.data), tuple(t_shape_g), qu)
                exp_mod.register_buffer(
                    "_gate_exp_packed",
                    torch.from_numpy(np.ascontiguousarray(gate_np)).to(torch.uint8),
                )
                exp_mod.register_buffer(
                    "_up_exp_packed",
                    torch.from_numpy(np.ascontiguousarray(up_np)).to(torch.uint8),
                )
                exp_mod.register_buffer("_gguf_qtype_gate", torch.tensor(qg, dtype=torch.int32))
                exp_mod.register_buffer("_gguf_qtype_up", torch.tensor(qu, dtype=torch.int32))
                loaded_count += 1
                de_t = tensor_map[de_key]
                qd = int(de_t.tensor_type)
                if not gguf_quant_type_supported_for_moe_packed(qd):
                    raise RuntimeError(
                        f"MoE packed GGUF requires quant types Q4_0/Q4_K/Q6_K; got down={qd}"
                    )
                down_np = numpy_gguf_data_to_packed_2d(np.asarray(de_t.data), tuple(t_shape_d), qd)
                exp_mod.register_buffer(
                    "_down_exp_packed",
                    torch.from_numpy(np.ascontiguousarray(down_np)).to(torch.uint8),
                )
                exp_mod.register_buffer("_gguf_qtype_down", torch.tensor(qd, dtype=torch.int32))
                loaded_count += 1
            elif getattr(exp_mod, "moe_cpu_offload", False) and getattr(exp_mod, "fp8_moe", False):
                t_shape_g = torch.Size([num_experts, moe_inter, hidden])
                t_shape_d = torch.Size([num_experts, hidden, moe_inter])
                dst_g = exp_mod._gate_up_fp8_cpu
                dst_gs = exp_mod._gate_up_scale_cpu
                dst_d = exp_mod._down_fp8_cpu
                dst_ds = exp_mod._down_scale_cpu
                ge = _dequantize_gguf_tensor(
                    tensor_map[ge_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ge_ok = ge is not None
                ue = _dequantize_gguf_tensor(
                    tensor_map[ue_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ue_ok = ue is not None
                if ge_ok and ue_ok:
                    br = moe_inter // moe_fp8_block_size()
                    for e in range(num_experts):
                        wf8_g, sg = fp8_block_quantize_2d(ge[e].contiguous())
                        wf8_u, su = fp8_block_quantize_2d(ue[e].contiguous())
                        dst_g[e, :moe_inter].copy_(wf8_g)
                        dst_g[e, moe_inter:].copy_(wf8_u)
                        dst_gs[e, :br].copy_(sg)
                        dst_gs[e, br:].copy_(su)
                    loaded_count += 1
                del ge, ue
                if de_key in tensor_map:
                    de = _dequantize_gguf_tensor(
                        tensor_map[de_key], "cpu", torch.float16, target_shape=t_shape_d
                    )
                    if de is not None:
                        for e in range(num_experts):
                            wf8, sf = fp8_block_quantize_2d(de[e].contiguous())
                            dst_d[e].copy_(wf8)
                            dst_ds[e].copy_(sf)
                        loaded_count += 1
                    del de
            elif getattr(exp_mod, "fp8_moe", False):
                t_shape_g = torch.Size([num_experts, moe_inter, hidden])
                t_shape_d = torch.Size([num_experts, hidden, moe_inter])
                dst = exp_mod.gate_up_proj.data
                dst_s = exp_mod.gate_up_scale.data
                dt_w = dst.dtype
                ge = _dequantize_gguf_tensor(
                    tensor_map[ge_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ge_ok = ge is not None
                ue = _dequantize_gguf_tensor(
                    tensor_map[ue_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ue_ok = ue is not None
                if ge_ok and ue_ok:
                    br = moe_inter // moe_fp8_block_size()
                    for e in range(num_experts):
                        wf8_g, sg = fp8_block_quantize_2d(ge[e].contiguous())
                        wf8_u, su = fp8_block_quantize_2d(ue[e].contiguous())
                        dst[e, :moe_inter].copy_(wf8_g.to(dtype=dt_w))
                        dst[e, moe_inter:].copy_(wf8_u.to(dtype=dt_w))
                        dst_s[e, :br].copy_(sg)
                        dst_s[e, br:].copy_(su)
                    loaded_count += 1
                del ge, ue
                if de_key in tensor_map:
                    de = _dequantize_gguf_tensor(
                        tensor_map[de_key], "cpu", torch.float16, target_shape=t_shape_d
                    )
                    if de is not None:
                        ddst = exp_mod.down_proj.data
                        ddst_s = exp_mod.down_scale.data
                        for e in range(num_experts):
                            wf8, sf = fp8_block_quantize_2d(de[e].contiguous())
                            ddst[e].copy_(wf8.to(dtype=ddst.dtype))
                            ddst_s[e].copy_(sf)
                        loaded_count += 1
                    del de
            elif hasattr(exp_mod, "gate_up_proj") and hasattr(exp_mod, "down_proj"):
                t_shape_g = torch.Size([num_experts, moe_inter, hidden])
                t_shape_d = torch.Size([num_experts, hidden, moe_inter])
                dst = exp_mod.gate_up_proj.data
                dt = dst.dtype
                # Dequant gate and up sequentially so ge and ue are never both resident (halves peak RSS).
                ge = _dequantize_gguf_tensor(
                    tensor_map[ge_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ge_ok = ge is not None
                if ge_ok:
                    dst[:, :moe_inter, :].copy_(ge.to(dtype=dt))
                del ge
                ue = _dequantize_gguf_tensor(
                    tensor_map[ue_key], "cpu", torch.float16, target_shape=t_shape_g
                )
                ue_ok = ue is not None
                if ue_ok:
                    dst[:, moe_inter:, :].copy_(ue.to(dtype=dt))
                del ue
                if ge_ok and ue_ok:
                    loaded_count += 1
                de = None
                if de_key in tensor_map:
                    de = _dequantize_gguf_tensor(
                        tensor_map[de_key], "cpu", torch.float16, target_shape=t_shape_d
                    )
                if de is not None:
                    exp_mod.down_proj.data.copy_(de.to(dtype=exp_mod.down_proj.dtype))
                    loaded_count += 1
                del de
        if getattr(hf_config, "model_type", "") == "qwen3_5_moe_text":
            gc.collect()
        elif i % 4 == 0:
            gc.collect()

    _has_shexp_gate = ("blk.0.ffn_gate_sw_shexp.weight" in tensor_map) or (
        "blk.0.ffn_gate_inp_shexp.weight" in tensor_map
    )
    if getattr(hf_config, "model_type", "") == "qwen3_5_moe_text" and not _has_shexp_gate:
        print(
            ">>> [Info] Qwen3.5 MoE GGUF: no blk.*.ffn_gate_sw_shexp / ffn_gate_inp_shexp; "
            "shared_expert_gate not loaded from GGUF (defaults retained)."
        )

    # Report unmapped GGUF tensors for debugging
    mapped_prefixes = set()
    for pat in map_rules:
        for i in range(num_layers):
            mapped_prefixes.add(pat.format(i=i))
    special_prefixes = set()
    for i in range(num_layers):
        for sp in ["ssm_a", "ssm_dt.bias", "ssm_conv1d.weight"]:
            special_prefixes.add(f"blk.{i}.{sp}")
        for sp in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            special_prefixes.add(f"blk.{i}.{sp}")
    all_mapped = mapped_prefixes | special_prefixes | {"token_embd.weight", "output_norm.weight", "output.weight"}
    unmapped = [n for n in tensor_map if n not in all_mapped]
    if unmapped:
        print(f">>> [Info] {len(unmapped)} GGUF tensors not mapped (may be MoE experts or unused): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")
    print(f">>> GGUF Weight loading complete. Loaded {loaded_count} parameters.")


def _dequantize_pack_quantized_int4_symmetric(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Match Neural Magic `pack_quantized` (HF compressed-tensors): unpack int32 nibbles
    then group dequantize. Falls back to local PyTorch if compressed-tensors is missing.
    """
    w32 = weight_packed.to(torch.int32)
    if weight_shape is not None and weight_shape.numel() >= 2:
        original = torch.Size(
            (int(weight_shape[0].item()), int(weight_shape[1].item()))
        )
    else:
        original = torch.Size((w32.shape[0], w32.shape[1] * 8))
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
            unpack_from_int32,
        )
        from compressed_tensors.quantization.lifecycle.forward import dequantize
    except ImportError:
        K = original[1]
        if weight_scale.shape[1] <= 0 or K % weight_scale.shape[1] != 0:
            raise RuntimeError(
                f"Bad scales shape {tuple(weight_scale.shape)} for in_features {K}"
            )
        group_size = K // weight_scale.shape[1]
        return dequantize_symmetric_packed_int4_pytorch(
            w32, weight_scale, group_size=group_size
        )

    unpacked = unpack_from_int32(w32, 4, original)
    return dequantize(
        unpacked.to(torch.float32),
        weight_scale.to(torch.float32),
        None,
        dtype=torch.float16,
    ).to(torch.float16)


def _checkpoint_has_language_model_layers(sd_keys: List[str]) -> bool:
    return any(k.startswith("model.language_model.layers.") for k in sd_keys)


def _fill_qwen35_moe_experts_fp8_from_accum(
    model: nn.Module, expert_accum: Dict[str, torch.Tensor]
) -> int:
    """
    Fill Qwen3.5 MoE expert FP8 CPU buffers from safetensors packed tensors.
    """
    if not expert_accum:
        return 0
    if not (qwen35_moe_fp8_enabled() and qwen35_moe_offload_enabled()):
        return 0
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is None:
        return 0
    inner = getattr(model, "model", None)
    hf_cfg = getattr(inner, "config", None) if inner is not None else None
    if hf_cfg is None:
        hf_cfg = getattr(model, "config", None)
    if hf_cfg is None or getattr(hf_cfg, "model_type", "") != "qwen3_5_moe_text":
        return 0
    n_layers = int(getattr(hf_cfg, "num_hidden_layers", 0))
    n_exp = int(getattr(hf_cfg, "num_experts", 0))
    moe_inter = int(getattr(hf_cfg, "moe_intermediate_size", 0))
    hidden = int(getattr(hf_cfg, "hidden_size", 0))
    if n_layers <= 0 or n_exp <= 0 or moe_inter <= 0 or hidden <= 0:
        return 0
    if not dims_ok_for_moe_fp8(hidden, moe_inter):
        return 0

    modules = dict(model.named_modules())
    br = moe_inter // moe_fp8_block_size()
    loaded_layers = 0

    def _get_proj_tensor(li: int, e: int, proj: str, suffix: str) -> Optional[torch.Tensor]:
        for prefix in (
            f"model.language_model.layers.{li}.mlp.experts.{e}.{proj}",
            f"model.layers.{li}.mlp.experts.{e}.{proj}",
        ):
            k = f"{prefix}.{suffix}"
            if k in expert_accum:
                return expert_accum[k]
        return None

    for li in range(n_layers):
        exp_mod = (
            modules.get(f"model.layers.{li}.mlp.experts")
            or modules.get(f"model.model.layers.{li}.mlp.experts")
        )
        if exp_mod is None or not getattr(exp_mod, "moe_cpu_offload", False):
            continue
        dst_g = getattr(exp_mod, "_gate_up_fp8_cpu", None)
        dst_gs = getattr(exp_mod, "_gate_up_scale_cpu", None)
        dst_d = getattr(exp_mod, "_down_fp8_cpu", None)
        dst_ds = getattr(exp_mod, "_down_scale_cpu", None)
        if dst_g is None or dst_gs is None or dst_d is None or dst_ds is None:
            continue

        ok_layer = True
        for e in range(n_exp):
            try:
                g_q = _get_proj_tensor(li, e, "gate_proj", "weight_packed")
                g_s = _get_proj_tensor(li, e, "gate_proj", "weight_scale")
                g_sh = _get_proj_tensor(li, e, "gate_proj", "weight_shape")
                u_q = _get_proj_tensor(li, e, "up_proj", "weight_packed")
                u_s = _get_proj_tensor(li, e, "up_proj", "weight_scale")
                u_sh = _get_proj_tensor(li, e, "up_proj", "weight_shape")
                d_q = _get_proj_tensor(li, e, "down_proj", "weight_packed")
                d_s = _get_proj_tensor(li, e, "down_proj", "weight_scale")
                d_sh = _get_proj_tensor(li, e, "down_proj", "weight_shape")
                if any(x is None for x in (g_q, g_s, u_q, u_s, d_q, d_s)):
                    ok_layer = False
                    break
                ge = _dequantize_pack_quantized_int4_symmetric(g_q, g_s, g_sh)
                ue = _dequantize_pack_quantized_int4_symmetric(u_q, u_s, u_sh)
                de = _dequantize_pack_quantized_int4_symmetric(d_q, d_s, d_sh)
                wf8_g, sg = fp8_block_quantize_2d(ge.contiguous())
                wf8_u, su = fp8_block_quantize_2d(ue.contiguous())
                dst_g[e, :moe_inter].copy_(wf8_g.to(dtype=dst_g.dtype))
                dst_g[e, moe_inter:].copy_(wf8_u.to(dtype=dst_g.dtype))
                dst_gs[e, :br].copy_(sg)
                dst_gs[e, br: 2 * br].copy_(su)
                wf8_d, sd = fp8_block_quantize_2d(de.contiguous())
                dst_d[e].copy_(wf8_d.to(dtype=dst_d.dtype))
                dst_ds[e].copy_(sd)
                del ge, ue, de, wf8_g, wf8_u, wf8_d, sg, su, sd
            except Exception as ex:
                print(f">>> [Warning] MoE expert FP8 fill failed layer={li} expert={e}: {ex}")
                ok_layer = False
                break
        if ok_layer:
            loaded_layers += 1
    if loaded_layers:
        print(
            f">>> Qwen3.5 MoE safetensors: filled FP8 CPU expert buffers for {loaded_layers}/{n_layers} layers."
        )
    return loaded_layers


def _safetensors_key_matches_main_model_layer(
    key: str, layer_idx: str, use_language_model_prefix: bool
) -> bool:
    """
    Avoid binding `mtp.layers.{i}.*` (or other aux modules) when the real weights live
    under `model.language_model.layers.{i}.*`.
    """
    needle = f"layers.{layer_idx}."
    if needle not in key:
        return False
    if use_language_model_prefix:
        return key.startswith("model.language_model.layers.")
    if key.startswith("mtp.") or key.startswith("model.mtp."):
        return False
    return True


def _param_copy_key_allowed(
    key: str, param_target: str, use_language_model_prefix: bool
) -> bool:
    """When copying dense params, skip auxiliary stacks that spoof `layers.{i}.`."""
    m = re.match(r"layers\.(\d+)\.", param_target)
    if m is None:
        return True
    return _safetensors_key_matches_main_model_layer(
        key, m.group(1), use_language_model_prefix
    )


def _looks_like_qwen35_35b_awq_model_path(model_path: str) -> bool:
    base = os.path.basename(os.path.abspath(model_path)).lower()
    return "qwen3.5-35b-awq" in base or ("qwen3.5" in base and "35b" in base and "awq" in base)


_QWEN35_35B_BALANCED_HIGH_FIDELITY_SUFFIXES = (
    ".linear_attn.in_proj_qkv",
    ".linear_attn.out_proj",
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".shared_expert.gate_proj",
    ".shared_expert.up_proj",
    ".shared_expert.down_proj",
)


def _should_force_high_fidelity_awq_for_qwen35_35b(module_name: str) -> bool:
    mode = os.environ.get(
        "FASTINFERENCE_QWEN35_35B_AWQ_HIGH_FIDELITY_MODE",
        "balanced",
    ).strip().lower()
    if mode in ("off", "0", "false", "no"):
        return False
    if mode in ("all", "full"):
        return ".experts." not in module_name
    return any(module_name.endswith(suffix) for suffix in _QWEN35_35B_BALANCED_HIGH_FIDELITY_SUFFIXES)


def _load_safetensors(model: nn.Module, model_path: str):
    from safetensors.torch import load_file
    sf_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    if not sf_files: return
    print(f">>> Loading Safetensors from {model_path}...")
    loaded_count = 0
    attr_map = {
        "qweight": ["weight_packed", "qweight", "weight"],
        "scales": ["weight_scale", "scales"],
        "qzeros": ["qzeros", "weight_zero"],
        "bias": ["bias"],
        "weight_shape": ["weight_shape"],
    }
    # nn.Linear (e.g. lm_head) also uses weight_packed in compressed-tensors checkpoints.
    # Do NOT match every submodule path containing "linear_attn": the parent `linear_attn`
    # nn.Module() container has no weights but used to pollute awq_accum and break loads.
    # Qwen3_5RMSNormGated / Conv1d under linear_attn load via params_dict (dense keys).
    lite_modules: Dict[str, nn.Module] = {}
    for n, mod in model.named_modules():
        if isinstance(mod, (LiteLinear, RMSNorm)):
            lite_modules[n] = mod
        elif isinstance(mod, nn.Linear):
            lite_modules[n] = mod
    params_dict = dict(model.named_parameters())
    loaded_params = set()
    # Sharded checkpoints split qweight / scales / qzeros across files; merge before dequant.
    awq_accum: Dict[str, Dict[str, torch.Tensor]] = {}
    collect_moe_expert = qwen35_moe_fp8_enabled() and qwen35_moe_offload_enabled()
    expert_accum: Dict[str, torch.Tensor] = {}
    for f in sf_files:
        print(f"    Processing {f}...")
        sd = load_file(os.path.join(model_path, f), device="cpu")
        if collect_moe_expert:
            for k, v in sd.items():
                if ".mlp.experts." in k and any(
                    x in k for x in ("weight_packed", "weight_scale", "weight_shape")
                ):
                    expert_accum[k] = v
        sd_keys = list(sd.keys())
        use_lm_layers = _checkpoint_has_language_model_layers(sd_keys)
        for m_name, m in lite_modules.items():
            idx_match = re.search(r"layers\.(\d+)\.", m_name)
            idx = idx_match.group(1) if idx_match else None
            proj = m_name.split(".")[-1]
            for internal, srcs in attr_map.items():
                found_v = None
                for s in srcs:
                    for k in sd_keys:
                        if idx is not None and f"layers.{idx}." in k:
                            if not _safetensors_key_matches_main_model_layer(
                                k, idx, use_lm_layers
                            ):
                                continue
                            if k.endswith(f".{proj}.{s}") or k.endswith(f".{proj.replace('self_attn', 'linear_attn')}.{s}"): found_v = sd[k]; break
                            if "linear_attn" in k and k.endswith(f".{s}") and proj in k:
                                if not any(x in k for x in ["packed", "scale", "zero"]): found_v = sd[k]; break
                        elif idx is None and "layers." not in k:
                            if k.endswith(f".{proj}.{s}") or k == f"{m_name}.{s}" or k.endswith(f".{m_name}.{s}"): found_v = sd[k]; break
                        if found_v is not None: break
                    if found_v is not None: break
                if found_v is None:
                    continue
                if internal == "bias":
                    if hasattr(m, "bias") and m.bias is not None: m.bias.data.copy_(found_v.to(dtype=m.bias.dtype))
                    else: m.bias = nn.Parameter(found_v.to(dtype=torch.float16))
                    loaded_count += 1
                elif internal == "qweight" and found_v.dtype in [torch.float16, torch.bfloat16]:
                    target_p = m.weight if hasattr(m, "weight") else m
                    if isinstance(target_p, nn.Parameter):
                        if target_p.numel() == 0 or target_p.shape != found_v.shape:
                            if hasattr(m, "weight"):
                                m.weight = nn.Parameter(found_v.to(dtype=torch.float16), requires_grad=False)
                        else:
                            target_p.data.copy_(found_v.to(dtype=target_p.dtype))
                        loaded_count += 1
                        loaded_params.add(m_name + ".weight")
                elif internal in ("qweight", "scales", "qzeros", "weight_shape"):
                    if m_name not in awq_accum:
                        awq_accum[m_name] = {}
                    awq_accum[m_name][internal] = found_v
        for p_name, p in params_dict.items():
            if p_name in loaded_params:
                continue
            target = p_name[6:] if p_name.startswith("model.") else p_name
            copied = False
            for k in sd_keys:
                if not _param_copy_key_allowed(k, target, use_lm_layers):
                    continue
                if k == target or k.endswith("." + target):
                    if p.shape == sd[k].shape:
                        p.data.copy_(sd[k].to(dtype=p.dtype))
                        loaded_count += 1
                        loaded_params.add(p_name)
                        copied = True
                        break
            if not copied and _try_load_legacy_flat_linear_attn_norm_from_safetensors(p_name, p, sd_keys, sd):
                loaded_count += 1
                loaded_params.add(p_name)
        del sd; gc.collect()

    for m_name, m in lite_modules.items():
        comps = awq_accum.get(m_name)
        if not comps or "qweight" not in comps or "scales" not in comps:
            continue
        qw = comps["qweight"]
        sc = comps["scales"]
        K = qw.shape[1] * 8
        if sc.shape[1] <= 0 or K % sc.shape[1] != 0:
            print(
                f">>> AWQ skip {m_name}: bad scales shape {tuple(sc.shape)} for packed in_features {K}"
            )
            continue
        G = K // sc.shape[1]
        try:
            quant_config = getattr(m, "quant_config", None)
            quant_name = (
                quant_config.get_name().lower()
                if quant_config is not None and hasattr(quant_config, "get_name")
                else ""
            )
            should_preserve_quant = isinstance(m, LiteLinear) and quant_name == "awq"

            if should_preserve_quant:
                if _looks_like_qwen35_35b_awq_model_path(model_path):
                    m.force_high_fidelity_awq = _should_force_high_fidelity_awq_for_qwen35_35b(
                        m_name
                    )
                m.qweight = nn.Parameter(qw.contiguous(), requires_grad=False)
                m.scales = nn.Parameter(sc.contiguous(), requires_grad=False)
                qz = comps.get("qzeros")
                m.qzeros = (
                    nn.Parameter(qz.contiguous(), requires_grad=False)
                    if qz is not None
                    else None
                )
                weight_shape = comps.get("weight_shape")
                m.weight_shape = (
                    tuple(int(x) for x in weight_shape.view(-1).tolist())
                    if weight_shape is not None
                    else None
                )
                m.group_size = G
                if hasattr(m, "_quant_weight"):
                    m._quant_weight = None
                loaded_count += 1
                loaded_params.add(m_name + ".qweight")
                continue

            if "qzeros" in comps:
                qz = comps["qzeros"]
                dq = dequantize_awq_pytorch(qw, sc, qz, group_size=G)
            else:
                dq = _dequantize_pack_quantized_int4_symmetric(
                    qw, sc, comps.get("weight_shape")
                )
            wparam = m.weight if hasattr(m, "weight") else None
            if wparam is None:
                continue
            if wparam.numel() == 0 or wparam.shape != dq.shape:
                m.weight = nn.Parameter(dq.to(dtype=torch.float16), requires_grad=False)
            else:
                m.weight.data.copy_(dq.to(dtype=torch.float16))
            if hasattr(m, "_quant_weight"):
                m._quant_weight = None
            loaded_count += 1
            loaded_params.add(m_name + ".weight")
        except Exception as e:
            print(f">>> AWQ dequant failed for {m_name}: {e}")

    for m_name, comps in awq_accum.items():
        has_q = "qweight" in comps
        has_s = "scales" in comps
        if has_q != has_s:
            print(
                f">>> [Warning] Incomplete AWQ shard merge for {m_name}: "
                f"keys={list(comps.keys())} (need both qweight and scales)"
            )

    _fill_qwen35_moe_experts_fp8_from_accum(model, expert_accum)
    if collect_moe_expert:
        del expert_accum
        gc.collect()

    print(f">>> Safetensors loading complete. Loaded {loaded_count} parameters.")

def _apply_cuda_then_hf_dtype(model: nn.Module, target_dtype: torch.dtype) -> None:
    """
    Move model to CUDA, then cast floating tensors to target_dtype without corrupting
    Qwen3.5 MoE FP8 expert weights or their float32 block scales.
    """
    model.to(device="cuda")
    f8 = getattr(torch, "float8_e4m3fn", None)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if f8 is not None and p.dtype == f8:
                continue
            if ".mlp.experts.gate_up_scale" in name or ".mlp.experts.down_scale" in name:
                continue
            if not p.is_floating_point():
                continue
            p.data = p.data.to(dtype=target_dtype)
        for name, b in model.named_buffers():
            if "_gate_up_fp8_cpu" in name or "_gate_up_scale_cpu" in name:
                continue
            if "_down_fp8_cpu" in name or "_down_scale_cpu" in name:
                continue
            if not b.is_floating_point():
                continue
            b.copy_(b.to(dtype=target_dtype))


def _resolve_model_dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    """
    Match Hugging Face checkpoint dtype from config (e.g. text_config.dtype / torch_dtype).
    Default float16 for older checkpoints without explicit dtype.
    """
    for attr in ("dtype", "torch_dtype"):
        ds = getattr(hf_config, attr, None)
        if isinstance(ds, str) and "bfloat16" in ds.lower():
            return torch.bfloat16
        if ds is torch.bfloat16:
            return torch.bfloat16
    return torch.float16


def get_tokenizer(model: Union[str, Any], **kwargs: Any):
    name = model.model if hasattr(model, "model") else model
    if hasattr(model, "trust_remote_code"): kwargs["trust_remote_code"] = model.trust_remote_code
    try: return AutoTokenizer.from_pretrained(name, **kwargs)
    except:
        try: return AutoTokenizer.from_pretrained(name, use_fast=False, **kwargs)
        except:
            class Dummy:
                def __init__(self): self.vocab_size = 154880; self.eos_token_id = 151329; self.pad_token_id = 151329
                def encode(self, t, **kwargs): 
                    ids = [1, 2, 3, 4, 5] * 2
                    if kwargs.get("return_tensors") == "pt": return torch.tensor([ids])
                    return ids
                def decode(self, ids, **kwargs): return " [Dummy Output] "
            return Dummy()

def get_model(vllm_config: VllmConfig) -> nn.Module:
    cfg = vllm_config.model_config
    if cfg.hf_config is None: cfg.hf_config = PretrainedConfig()
    path = os.path.join(cfg.model, "config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                if k != "text_config": setattr(cfg.hf_config, k, v)
            if "text_config" in data: 
                for k, v in data["text_config"].items(): setattr(cfg.hf_config, k, v)
            cfg.hf_config.layer_types = data.get("text_config", {}).get("layer_types", [])
    if getattr(cfg.hf_config, "model_type", "") == "deepseek_v2":
        setattr(cfg.hf_config, "num_key_value_heads", getattr(cfg.hf_config, "num_attention_heads", 40))
        setattr(cfg.hf_config, "head_dim", 128)
    model_cls, _ = ModelRegistry.resolve_model_cls(getattr(cfg.hf_config, "architectures", ["LlamaForCausalLM"]), cfg)
    model = model_cls(vllm_config)
    gguf_files = [f for f in os.listdir(cfg.model) if f.endswith(".gguf")]
    if gguf_files:
        _load_gguf_weights(model, os.path.join(cfg.model, gguf_files[0]), cfg.hf_config)
        # Match HF text dtype (Qwen3.5 is usually bfloat16); dequant uses fp16/bf16 tensors.
        target_dtype = _resolve_model_dtype_from_hf_config(cfg.hf_config)
    else:
        _load_safetensors(model, cfg.model)
        target_dtype = _resolve_model_dtype_from_hf_config(cfg.hf_config)
    print(f">>> Moving model to CUDA (dtype={target_dtype})...")
    _apply_cuda_then_hf_dtype(model, target_dtype)
    return model.eval()
