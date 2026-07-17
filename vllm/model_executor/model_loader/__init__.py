# SPDX-License-Identifier: Apache-2.0
import contextlib
import gc
import json
import os
import re
from math import prod
from typing import Any

import gguf
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_awq_pytorch,
    dequantize_symmetric_packed_int4_pytorch,
)
from vllm.model_executor.model_loader.profile_hints import (
    awq_profile_hint_from_model_path as _awq_profile_hint_from_model_path,
)
from vllm.model_executor.model_loader.profile_hints import (
    looks_like_gemma4_31b_model_path as _looks_like_gemma4_31b_model_path,
)
from vllm.model_executor.model_loader.profile_hints import (
    looks_like_qwen35_9b_awq_model_path as _looks_like_qwen35_9b_awq_model_path,
)
from vllm.model_executor.model_loader.profile_hints import (
    qwen35_awq_profile_hint_from_model_path as _qwen35_awq_profile_hint_from_model_path,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.moe_fp8_utils import (
    dims_ok_for_moe_fp8,
    fp8_block_quantize_2d,
    moe_fp8_block_size,
    moe_fp8_enabled,
    moe_offload_enabled,
)
from vllm.model_executor.moe_gguf_packed import (
    gguf_quant_type_supported_for_moe_packed,
    numpy_gguf_data_to_packed_2d,
)
from vllm.transformers_utils.configs.gemma4 import build_fallback_hf_config


def _dequantize_gguf_tensor(
    tensor: Any,
    device: str | torch.device,
    dtype: torch.dtype,
    target_shape: torch.Size | None = None,
) -> torch.Tensor | None:
    """Materialize a GGUF tensor for the maintained local loader."""
    try:
        quant_type = gguf.GGMLQuantizationType(int(tensor.tensor_type))
        values = gguf.dequantize(np.asarray(tensor.data), quant_type)
        result = torch.from_numpy(np.array(values, copy=True))
        if target_shape is not None and result.numel() == prod(target_shape):
            result = result.reshape(target_shape)
        return result.to(device=device, dtype=dtype)
    except Exception:
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
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1 :]
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
    return _reorder_v_heads_ggml_to_hf(
        source.unsqueeze(-1), 0, num_k, num_v_per_k, 1
    ).squeeze(-1)


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
    sd_keys: list[str],
    sd: dict[str, torch.Tensor],
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
    hf_config: Any | None = None,
) -> None:
    actual_param = param
    logical_shape = None
    if isinstance(param, LiteLinear):
        logical_shape = torch.Size([param.output_size, param.input_size])
        actual_param = param.weight
    elif isinstance(param, nn.Module) and hasattr(param, "weight"):
        actual_param = param.weight
    if not isinstance(actual_param, torch.Tensor) and logical_shape is None:
        return
    target_shape = logical_shape if logical_shape is not None else actual_param.shape
    # Use universal dequant via gguf library
    source = _dequantize_gguf_tensor(gguf_tensor, "cpu", torch.float16, target_shape)
    if source is None:
        return

    tname = str(getattr(gguf_tensor, "name", "") or "")
    if hf_config is not None and any(
        x in tname
        for x in (
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_out.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
        )
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
            param.weight = nn.Parameter(
                source.to(dtype=torch.float16), requires_grad=False
            )
        else:
            param.weight.data.copy_(source.to(dtype=param.weight.dtype))
    if isinstance(actual_param, nn.Parameter):
        # Parameter update
        if actual_param.numel() == 0:
            # Resize empty parameter
            actual_param.data = source.to(dtype=actual_param.dtype)
            print(
                f"    [Verify] {gguf_tensor.name} resized to {actual_param.shape}, numel={actual_param.numel()}"
            )
        elif actual_param.shape == source.shape:
            actual_param.data.copy_(source)
        else:
            # Heuristic mapping (legacy); prefer correct layout from _dequantize_gguf_tensor.
            try:
                o, i = actual_param.shape[0], actual_param.shape[1]
                if source.ndim == 2 and source.shape == (i, o):
                    actual_param.data.copy_(source.T.contiguous())
                elif (
                    source.shape[0] == actual_param.shape[1]
                    and source.shape[1] >= actual_param.shape[0]
                ):
                    actual_param.data.copy_(source[:, : actual_param.shape[0]].T)
                elif (
                    source.shape[1] == actual_param.shape[1]
                    and source.shape[0] >= actual_param.shape[0]
                ):
                    actual_param.data.copy_(source[: actual_param.shape[0], :])
            except Exception:
                pass
    elif isinstance(actual_param, torch.Tensor) and actual_param.shape == source.shape:
        actual_param.data.copy_(source)
    with contextlib.suppress(NameError):
        del source


def _dequantize_pack_quantized_int4_symmetric(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor | None,
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
            ) from None
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


def _checkpoint_has_language_model_layers(sd_keys: list[str]) -> bool:
    return any(k.startswith("model.language_model.layers.") for k in sd_keys)


def _fill_moe_experts_fp8_from_accum(
    model: nn.Module, expert_accum: dict[str, torch.Tensor]
) -> int:
    """
    Fill MoE expert FP8 CPU buffers from safetensors packed tensors.
    """
    if not expert_accum:
        return 0
    if not (moe_fp8_enabled() and moe_offload_enabled()):
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

    def _get_proj_tensor(
        li: int, e: int, proj: str, suffix: str
    ) -> torch.Tensor | None:
        for prefix in (
            f"model.language_model.layers.{li}.mlp.experts.{e}.{proj}",
            f"model.layers.{li}.mlp.experts.{e}.{proj}",
        ):
            k = f"{prefix}.{suffix}"
            if k in expert_accum:
                return expert_accum[k]
        return None

    for li in range(n_layers):
        exp_mod = modules.get(f"model.layers.{li}.mlp.experts") or modules.get(
            f"model.model.layers.{li}.mlp.experts"
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
                dst_gs[e, br : 2 * br].copy_(su)
                wf8_d, sd = fp8_block_quantize_2d(de.contiguous())
                dst_d[e].copy_(wf8_d.to(dtype=dst_d.dtype))
                dst_ds[e].copy_(sd)
                del ge, ue, de, wf8_g, wf8_u, wf8_d, sg, su, sd
            except Exception as ex:
                print(
                    f">>> [Warning] MoE expert FP8 fill failed layer={li} expert={e}: {ex}"
                )
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
    return not (key.startswith("mtp.") or key.startswith("model.mtp."))


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


def _looks_like_hf_repo_id(model_ref: str) -> bool:
    return (
        isinstance(model_ref, str)
        and ("/" in model_ref)
        and (not os.path.isabs(model_ref))
    )


def _resolve_model_dir_for_weights(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    if not _looks_like_hf_repo_id(model_path):
        return model_path
    try:
        return snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model repo {model_path!r}: {e}") from e


def _load_safetensors(
    model: nn.Module, model_path: str, target_dtype: torch.dtype = torch.float16
):
    from safetensors.torch import load_file

    local_model_dir = _resolve_model_dir_for_weights(model_path)
    sf_files = sorted(
        [f for f in os.listdir(local_model_dir) if f.endswith(".safetensors")]
    )
    if not sf_files:
        return
    print(
        f">>> Loading Safetensors from {local_model_dir} (Casting to {target_dtype} on-the-fly)..."
    )
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
    lite_modules: dict[str, nn.Module] = {}
    for n, mod in model.named_modules():
        if isinstance(mod, (LiteLinear, RMSNorm, nn.Linear)):
            lite_modules[n] = mod
    params_dict = dict(model.named_parameters())
    loaded_params = set()
    loaded_module_attrs: set[str] = set()
    # Sharded checkpoints split qweight / scales / qzeros across files; merge before dequant.
    awq_accum: dict[str, dict[str, torch.Tensor]] = {}
    collect_moe_expert = moe_fp8_enabled() and moe_offload_enabled()
    expert_accum: dict[str, torch.Tensor] = {}
    seen_language_model_layers = False
    for f in sf_files:
        print(f"    Processing {f}...")
        sd = load_file(os.path.join(local_model_dir, f), device="cpu")
        if collect_moe_expert:
            for k, v in sd.items():
                if ".mlp.experts." in k and any(
                    x in k for x in ("weight_packed", "weight_scale", "weight_shape")
                ):
                    expert_accum[k] = v
        sd_keys = list(sd.keys())
        if _checkpoint_has_language_model_layers(sd_keys):
            seen_language_model_layers = True
        # Keep language-model prefix matching sticky across shards.
        # Multimodal checkpoints can have tail shards that only contain
        # vision tensors; falling back to loose "layers.{i}" matching there
        # can incorrectly overwrite already-loaded text-layer params.
        use_lm_layers = seen_language_model_layers
        for m_name, m in lite_modules.items():
            idx_match = re.search(r"layers\.(\d+)\.", m_name)
            idx = idx_match.group(1) if idx_match else None
            proj = m_name.split(".")[-1]
            for internal, srcs in attr_map.items():
                module_internal_key = f"{m_name}.{internal}"
                if module_internal_key in loaded_module_attrs:
                    continue
                found_v = None
                for s in srcs:
                    alt_suffixes = [s]
                    if s == "weight_packed":
                        alt_suffixes.append("packed")
                    elif s == "weight_scale":
                        alt_suffixes.append("scale")
                    elif s == "weight_shape":
                        alt_suffixes.append("shape")
                    elif s == "weight_zero":
                        alt_suffixes.append("zero")
                    for k in sd_keys:
                        if k not in sd:
                            continue
                        if idx is not None and f"layers.{idx}." in k:
                            if not _safetensors_key_matches_main_model_layer(
                                k, idx, use_lm_layers
                            ):
                                continue
                            matched = False
                            for suffix in alt_suffixes:
                                if (
                                    k.endswith(f".{proj}.{suffix}")
                                    or k.endswith(f".{proj}_{suffix}")
                                    or k.endswith(
                                        f".{proj.replace('self_attn', 'linear_attn')}.{suffix}"
                                    )
                                    or k.endswith(
                                        f".{proj.replace('self_attn', 'linear_attn')}_{suffix}"
                                    )
                                ):
                                    matched = True
                                    break
                            if matched:
                                found_v = sd[k]
                                break
                            if (
                                "linear_attn" in k
                                and k.endswith(f".{s}")
                                and proj in k
                                and not any(x in k for x in ["packed", "scale", "zero"])
                            ):
                                found_v = sd[k]
                                break
                        elif idx is None:
                            matched_key = _find_non_layer_module_attr_key(
                                sd_keys,
                                m_name=m_name,
                                proj=proj,
                                alt_suffixes=alt_suffixes,
                            )
                            if matched_key is not None and matched_key in sd:
                                k = matched_key
                                found_v = sd[k]
                                break
                        if found_v is not None:
                            break
                    if found_v is not None:
                        break
                if found_v is None:
                    continue
                if internal == "bias":
                    # [35B SURVIVAL] Move module shell to GPU before assigning parameter
                    m.to("cuda")
                    m.bias = nn.Parameter(
                        found_v.to(device="cuda", dtype=target_dtype),
                        requires_grad=False,
                    )
                    loaded_count += 1
                    loaded_module_attrs.add(module_internal_key)
                    del sd[k]
                    torch.cuda.empty_cache()
                elif internal == "qweight" and _is_dense_safetensors_weight(found_v):
                    m.to("cuda")
                    if hasattr(m, "weight"):
                        m.weight = None
                    m.weight = nn.Parameter(
                        found_v.to(device="cuda", dtype=target_dtype),
                        requires_grad=False,
                    )
                    loaded_count += 1
                    loaded_params.add(m_name + ".weight")
                    loaded_module_attrs.add(module_internal_key)
                    del sd[k]
                    torch.cuda.empty_cache()
                elif internal in ("qweight", "scales", "qzeros", "weight_shape"):
                    m.to("cuda")
                    if hasattr(m, internal):
                        setattr(m, internal, None)
                    if m_name not in awq_accum:
                        awq_accum[m_name] = {}
                    if internal in awq_accum[m_name]:
                        continue
                    awq_accum[m_name][internal] = found_v.to(device="cuda")
                    loaded_module_attrs.add(module_internal_key)
                    del sd[k]
                    torch.cuda.empty_cache()
        for p_name, p in params_dict.items():
            if p_name in loaded_params:
                continue
            target = p_name[6:] if p_name.startswith("model.") else p_name
            target_aliases = {target}
            if ".mlp.router." in target:
                target_aliases.add(target.replace(".mlp.router.", ".router."))
            if ".mlp.experts." in target:
                target_aliases.add(target.replace(".mlp.experts.", ".experts."))
            copied = False
            for k in list(sd.keys()):
                if not any(
                    _param_copy_key_allowed(k, cand, use_lm_layers)
                    for cand in target_aliases
                ):
                    continue
                if any(k == cand or k.endswith("." + cand) for cand in target_aliases):
                    parts = p_name.split(".")
                    obj = model
                    try:
                        for part in parts[:-1]:
                            if hasattr(obj, part):
                                obj = getattr(obj, part)
                            elif (
                                isinstance(obj, (nn.ModuleList, nn.ParameterList, list))
                                and part.isdigit()
                            ):
                                obj = obj[int(part)]
                            else:
                                obj = getattr(obj, part)

                        # Move parent object to GPU before assignment
                        if isinstance(obj, nn.Module):
                            obj.to("cuda")

                        val_on_gpu = nn.Parameter(
                            sd[k].to(device="cuda", dtype=target_dtype),
                            requires_grad=False,
                        )
                        if isinstance(obj, (nn.ModuleList, nn.ParameterList, list)):
                            obj[int(parts[-1])] = val_on_gpu
                        else:
                            setattr(obj, parts[-1], val_on_gpu)

                        loaded_count += 1
                        loaded_params.add(p_name)
                        copied = True
                        del sd[k]
                        torch.cuda.empty_cache()
                        break
                    except Exception:
                        continue
            if not copied and _try_load_legacy_flat_linear_attn_norm_from_safetensors(
                p_name, p, list(sd.keys()), sd
            ):
                loaded_count += 1
                loaded_params.add(p_name)
        del sd
        gc.collect()
        torch.cuda.empty_cache()

    for m_name, m in lite_modules.items():
        comps = awq_accum.get(m_name)
        if not comps or "qweight" not in comps or "scales" not in comps:
            continue
        qw = comps["qweight"]
        sc = comps["scales"]
        packed_cols = qw.shape[-1]
        K = packed_cols * 8
        scale_groups = sc.shape[-1] if sc.ndim >= 2 else 0
        if scale_groups <= 0 or K % scale_groups != 0:
            print(
                f">>> AWQ skip {m_name}: bad scales shape {tuple(sc.shape)} for packed in_features {K}"
            )
            continue
        G = K // scale_groups
        try:
            # Force preserve quant for all AWQ models to avoid CPU OOM
            m.awq_profile_hint = _awq_profile_hint_from_model_path(model_path)

            # Tensors are already on GPU from immediate transfer above
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
        except Exception as e:
            print(f">>> AWQ load failed for {m_name}: {e}")

    for m_name, comps in awq_accum.items():
        has_q = "qweight" in comps
        has_s = "scales" in comps
        if has_q != has_s:
            print(
                f">>> [Warning] Incomplete AWQ shard merge for {m_name}: "
                f"keys={list(comps.keys())} (need both qweight and scales)"
            )

    _fill_moe_experts_fp8_from_accum(model, expert_accum)
    if collect_moe_expert:
        del expert_accum
        gc.collect()

    print(f">>> Safetensors loading complete. Loaded {loaded_count} parameters.")


def _apply_cuda_then_hf_dtype(model: nn.Module, target_dtype: torch.dtype) -> None:
    """
    Move model to CUDA, then cast floating tensors to target_dtype without corrupting
    Qwen3.5 MoE FP8 expert weights or their float32 block scales.
    """
    # [35B FIX] We NO LONGER call model.to("cuda") globally here because _load_safetensors
    # already moved them layer-by-layer. Calling it again on 35B causes OOM due to peak allocation.
    f8 = getattr(torch, "float8_e4m3fn", None)
    filter_suffixes = (
        ".mlp.experts.gate_up_scale",
        ".mlp.experts.down_scale",
        "_gate_up_fp8_cpu",
        "_gate_up_scale_cpu",
        "_down_fp8_cpu",
        "_down_scale_cpu",
    )

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.device.type != "cuda":
                p.data = p.data.to(device="cuda")
            if f8 is not None and p.dtype == f8:
                continue
            if any(name.endswith(s) for s in filter_suffixes):
                continue
            if not p.is_floating_point():
                continue
            if p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)

        for name, b in model.named_buffers():
            if b.device.type != "cuda":
                # Some buffers might be on CPU, move them
                b.data = b.data.to(device="cuda")
            if any(name.endswith(s) for s in filter_suffixes):
                continue
            if not b.is_floating_point():
                continue
            if b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)

    torch.cuda.empty_cache()


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


def _find_non_layer_module_attr_key(
    sd_keys: list[str],
    *,
    m_name: str,
    proj: str,
    alt_suffixes: list[str],
) -> str | None:
    exact_names = [m_name]
    if m_name.startswith("model."):
        exact_names.append(f"model.language_model.{m_name[6:]}")

    def exact_match(k: str, suffix: str) -> bool:
        return any(
            k == f"{name}.{suffix}"
            or k.endswith(f".{name}.{suffix}")
            or k == f"{name}_{suffix}"
            or k.endswith(f".{name}_{suffix}")
            for name in exact_names
        )

    def loose_match(k: str, suffix: str) -> bool:
        return k.endswith(f".{proj}.{suffix}") or k.endswith(f".{proj}_{suffix}")

    non_layer_keys = [k for k in sd_keys if "layers." not in k]
    for suffix in alt_suffixes:
        for k in non_layer_keys:
            if exact_match(k, suffix):
                return k
    for suffix in alt_suffixes:
        for k in non_layer_keys:
            if loose_match(k, suffix):
                return k
    return None


def _is_dense_safetensors_weight(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point()


def get_tokenizer(model: str | Any, **kwargs: Any):
    name = model.model if hasattr(model, "model") else model
    if hasattr(model, "trust_remote_code"):
        kwargs["trust_remote_code"] = model.trust_remote_code
    if hasattr(model, "revision") and model.revision is not None:
        kwargs["revision"] = model.revision
    try:
        return AutoTokenizer.from_pretrained(name, **kwargs)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(name, use_fast=False, **kwargs)
        except Exception:

            class Dummy:
                def __init__(self):
                    self.vocab_size = 154880
                    self.eos_token_id = 151329
                    self.pad_token_id = 151329

                def encode(self, t, **kwargs):
                    ids = [1, 2, 3, 4, 5] * 2
                    if kwargs.get("return_tensors") == "pt":
                        return torch.tensor([ids])
                    return ids

                def decode(self, ids, **kwargs):
                    return " [Dummy Output] "

            return Dummy()


def _maybe_init_quant_config_from_hf(vllm_config: VllmConfig) -> None:
    if getattr(vllm_config, "quant_config", None) is not None:
        return
    model_cfg = vllm_config.model_config
    hf_cfg = getattr(model_cfg, "hf_config", None)
    if hf_cfg is None:
        return
    quant_cfg = getattr(hf_cfg, "quantization_config", None)
    if quant_cfg is None:
        text_cfg = getattr(hf_cfg, "text_config", None)
        if text_cfg is not None:
            quant_cfg = getattr(text_cfg, "quantization_config", None)
    if quant_cfg is None:
        quant_cfg = getattr(hf_cfg, "compression_config", None)
    if not isinstance(quant_cfg, dict):
        return

    quant_method = str(quant_cfg.get("quant_method", "")).lower()
    if quant_method == "awq":
        vllm_config.quant_config = AWQConfig.from_config(quant_cfg)
    elif quant_method == "compressed-tensors":
        vllm_config.quant_config = CompressedTensorsConfig.from_config(quant_cfg)


def _should_skip_safetensors_load(vllm_config: VllmConfig) -> bool:
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    archs = getattr(hf_config, "architectures", []) or []
    return model_type in ("deepseek_v4", "deepseek4", "deepseek_v4_flash") and any(
        "deepseekv4flash" in str(arch).lower() for arch in archs
    )


def _build_deepseek_v4_flash_inspect_model(
    model_cls: type[nn.Module],
    vllm_config: VllmConfig,
) -> nn.Module:
    from vllm.model_executor.models.deepseek_v4_flash.config import (
        DeepSeekV4FlashMemoryPolicy,
    )
    from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
        DeepSeekV4FlashGPUBackend,
        DeepSeekV4FlashGPUCapabilities,
    )
    from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
        open_deepseek_v4_flash_weight_store,
    )

    cfg = vllm_config.model_config
    store = open_deepseek_v4_flash_weight_store(cfg.model)
    try:
        policy = DeepSeekV4FlashMemoryPolicy()
        budget = policy.estimate_runtime_budget(
            getattr(cfg, "max_model_len", policy.max_first_release_context),
            model_mmap_bytes=store.diagnostics.file_size_bytes,
        )
        policy.validate_runtime_budget(budget)
        backend = DeepSeekV4FlashGPUBackend(
            capabilities=DeepSeekV4FlashGPUCapabilities(
                q8_linear=True,
                attention=True,
                compressed_attention=True,
                cache_update=True,
                moe=True,
                output=True,
            )
        )
        model = model_cls(vllm_config, gpu_backend=backend)
        model_with_store: Any = model
        model_with_store.attach_weight_store(store, budget)
        return model.eval()
    except Exception:
        store.close()
        raise


def get_model(vllm_config: VllmConfig) -> nn.Module:
    cfg = vllm_config.model_config
    if cfg.hf_config is None:
        cfg.hf_config = PretrainedConfig()
    path = os.path.join(cfg.model, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            cfg.hf_config = build_fallback_hf_config(data)
    elif _should_skip_safetensors_load(vllm_config):
        pass
    elif _looks_like_hf_repo_id(cfg.model):
        try:
            hf_auto_cfg = AutoConfig.from_pretrained(
                cfg.model,
                trust_remote_code=cfg.trust_remote_code,
                revision=cfg.revision,
            )
            cfg.hf_config = build_fallback_hf_config(hf_auto_cfg.to_dict())
        except Exception as e:
            print(
                f">>> [Warning] AutoConfig.from_pretrained failed for {cfg.model}: {e}"
            )
    _maybe_init_quant_config_from_hf(vllm_config)
    if getattr(cfg.hf_config, "model_type", "") == "deepseek_v2":
        cfg.hf_config.num_key_value_heads = getattr(
            cfg.hf_config, "num_attention_heads", 40
        )
        cfg.hf_config.head_dim = 128
    model_cls, _ = ModelRegistry.resolve_model_cls(
        getattr(cfg.hf_config, "architectures", ["LlamaForCausalLM"]), cfg
    )
    if _should_skip_safetensors_load(vllm_config):
        return _build_deepseek_v4_flash_inspect_model(model_cls, vllm_config)
    model = model_cls(vllm_config)

    # Pre-resolve target dtype
    target_dtype = _resolve_model_dtype_from_hf_config(cfg.hf_config)

    # [35B FIX] Pre-cast to target dtype on CPU. Fast and memory-safe for empty tensors.
    model.to(dtype=target_dtype)

    # [35B OOM FIX] Atomic loading to avoid peaks.
    if not _should_skip_safetensors_load(vllm_config):
        _load_safetensors(model, cfg.model, target_dtype=target_dtype)

    # [35B STABILITY] Final deterministic device and dtype sync.
    # Essential for rotary_emb caches, missed expert parameters, etc.
    print(">>> Synchronizing all model parameters to CUDA...")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.device.type != "cuda":
                p.data = p.data.to(device="cuda", dtype=target_dtype)
                torch.cuda.empty_cache()
        for n, b in model.named_buffers():
            if b.device.type != "cuda":
                b.data = b.data.to(device="cuda")
                torch.cuda.empty_cache()

    # Final deterministic cast for consistency
    if target_dtype == torch.bfloat16:
        model.bfloat16()
    elif target_dtype == torch.float16:
        model.half()

    torch.cuda.empty_cache()
    return model.eval()
