# SPDX-License-Identifier: Apache-2.0
from typing import Any
import os

import gguf
import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.gguf_kernels import ggml_dequantize_fallback
from vllm.model_executor.models.registry import ModelRegistry


def _resolve_target_dtype(vllm_config: VllmConfig) -> torch.dtype:
    configured_dtype = getattr(vllm_config.model_config, "dtype", None)
    if isinstance(configured_dtype, torch.dtype):
        return configured_dtype
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    hf_dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
    if isinstance(hf_dtype, torch.dtype):
        return hf_dtype
    if isinstance(hf_dtype, str):
        normalized = hf_dtype.lower()
        if "bfloat16" in normalized or normalized == "bf16":
            return torch.bfloat16
        if "float16" in normalized or normalized == "fp16":
            return torch.float16
        if "float32" in normalized or normalized == "fp32":
            return torch.float32
    return torch.float16


def _resolve_hf_dtype(hf_config: Any) -> torch.dtype:
    hf_dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
    if isinstance(hf_dtype, torch.dtype):
        return hf_dtype
    if isinstance(hf_dtype, str):
        normalized = hf_dtype.lower()
        if "bfloat16" in normalized or normalized == "bf16":
            return torch.bfloat16
        if "float32" in normalized or normalized == "fp32":
            return torch.float32
    return torch.float16


def _is_qwen3_5_dense(hf_config: Any) -> bool:
    architectures = getattr(hf_config, "architectures", [])
    is_qwen3_5 = any("Qwen3_5" in arch for arch in architectures)
    return is_qwen3_5 and int(getattr(hf_config, "num_experts", 0)) == 0


def _is_qwen9_aggressive_enabled(hf_config: Any) -> bool:
    if not _is_qwen3_5_dense(hf_config):
        return False
    if os.environ.get("FASTINFERENCE_QWEN9_STABLE", "0") == "1":
        return False
    # Default-on for Qwen3.5-9B dense path; user can disable via
    # FASTINFERENCE_QWEN9_AGGRESSIVE=0 or FASTINFERENCE_QWEN9_STABLE=1.
    return os.environ.get("FASTINFERENCE_QWEN9_AGGRESSIVE", "1") == "1"


def _is_deepseek_aggressive_enabled(hf_config: Any) -> bool:
    architectures = getattr(hf_config, "architectures", [])
    is_ds = any("DeepseekV2" in arch or "DeepseekV3" in arch for arch in architectures)
    if not is_ds:
        return False
    # DeepSeek-V2-Lite is relatively small (16B), we can afford aggressive fp16
    # if it's not explicitly disabled.
    if os.environ.get("FASTINFERENCE_DEEPSEEK_STABLE", "0") == "1":
        return False
    return os.environ.get("FASTINFERENCE_DEEPSEEK_AGGRESSIVE", "1") == "1"


def _to_torch_tensor(gguf_tensor: Any, device: str = "cuda") -> torch.Tensor:
    return torch.from_numpy(np.array(gguf_tensor.data, copy=True)).to(device=device)


def _load_param_from_gguf_tensor(param: torch.nn.Parameter, gguf_tensor: Any) -> None:
    source = _dequantize_gguf_tensor(gguf_tensor, str(param.device), param.dtype)
    if source.shape == param.shape:
        param.data.copy_(source)
        return
    if source.t().shape == param.shape:
        param.data.copy_(source.t().contiguous())
        return
    raise RuntimeError(f"Cannot map GGUF tensor into parameter shape: {source.shape} -> {param.shape}")


def _dequantize_gguf_tensor(gguf_tensor: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    quant_type = int(gguf_tensor.tensor_type)
    if quant_type in (0, 1):
        return _to_torch_tensor(gguf_tensor, device=device).to(dtype=dtype)
    packed = _to_torch_tensor(gguf_tensor, device=device)
    if packed.dim() == 3:
        # For many GGUF expert/MLA tensors, packed shape is [groups, rows, bytes].
        # ggml_dequantize_fallback expects `m` to be per-group rows.
        rows = int(packed.shape[1])
    else:
        rows = int(packed.shape[0])
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    cols = int((packed.shape[-1] // type_size) * block_size)
    return ggml_dequantize_fallback(packed, quant_type, rows, cols, dtype)


def _read_gguf_tensor_map_tolerant(gguf_file: str) -> tuple[dict[str, Any], list[str]]:
    try:
        reader = gguf.GGUFReader(gguf_file)
        return {t.name: t for t in reader.tensors}, []
    except Exception:
        # Fallback reader: keep parsing tensor metadata and skip tensors that
        # cannot be materialized by the current gguf Python package.
        import gguf.gguf_reader as gr
        from collections import OrderedDict as _OrderedDict

        reader = gr.GGUFReader.__new__(gr.GGUFReader)
        reader.data = np.memmap(gguf_file, mode="r")
        reader.byte_order = "I"
        reader.alignment = 32
        reader.fields = _OrderedDict()
        reader.tensors = []
        offs = 0
        if reader._get(offs, np.uint32, override_order="<")[0] != gr.GGUF_MAGIC:
            raise RuntimeError(f"GGUF magic invalid for {gguf_file}")
        offs += 4
        temp_version = reader._get(offs, np.uint32)
        if temp_version[0] & 65535 == 0:
            reader.byte_order = "S"
            temp_version = temp_version.view(temp_version.dtype.newbyteorder(reader.byte_order))
        offs += reader._push_field(gr.ReaderField(offs, "GGUF.version", [temp_version], [0], [gr.GGUFValueType.UINT32]))
        temp_counts = reader._get(offs, np.uint64, 2)
        offs += reader._push_field(gr.ReaderField(offs, "GGUF.tensor_count", [temp_counts[:1]], [0], [gr.GGUFValueType.UINT64]))
        offs += reader._push_field(gr.ReaderField(offs, "GGUF.kv_count", [temp_counts[1:]], [0], [gr.GGUFValueType.UINT64]))
        tensor_count, kv_count = temp_counts
        offs = reader._build_fields(offs, kv_count)
        offs, tensor_fields = reader._build_tensor_info(offs, tensor_count)
        new_align = reader.fields.get("general.alignment")
        if new_align is not None and new_align.types == [gr.GGUFValueType.UINT32]:
            reader.alignment = new_align.parts[-1][0]
        padding = offs % reader.alignment
        if padding != 0:
            offs += reader.alignment - padding

        tensor_map: dict[str, Any] = {}
        skipped: list[str] = []
        for field in tensor_fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding="utf-8")
            ggml_type = gr.GGMLQuantizationType(raw_dtype[0])
            n_elems = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            block_size, type_size = gr.GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(offs + offset_tensor[0])

            if ggml_type == gr.GGMLQuantizationType.F16:
                item_count, item_type = n_elems, np.float16
            elif ggml_type == gr.GGMLQuantizationType.F32:
                item_count, item_type = n_elems, np.float32
            elif ggml_type == gr.GGMLQuantizationType.F64:
                item_count, item_type = n_elems, np.float64
            elif ggml_type == gr.GGMLQuantizationType.I8:
                item_count, item_type = n_elems, np.int8
            elif ggml_type == gr.GGMLQuantizationType.I16:
                item_count, item_type = n_elems, np.int16
            elif ggml_type == gr.GGMLQuantizationType.I32:
                item_count, item_type = n_elems, np.int32
            elif ggml_type == gr.GGMLQuantizationType.I64:
                item_count, item_type = n_elems, np.int64
            else:
                item_count, item_type = n_bytes, np.uint8
                np_dims = gr.quant_shape_to_byte_shape(np_dims, ggml_type)
            try:
                data = reader._get(data_offs, item_type, item_count).reshape(np_dims)
            except Exception:
                skipped.append(tensor_name)
                continue
            tensor_map[tensor_name] = gr.ReaderTensor(
                name=tensor_name,
                tensor_type=ggml_type,
                shape=dims,
                n_elements=n_elems,
                n_bytes=n_bytes,
                data_offset=data_offs,
                data=data,
                field=field,
            )
        return tensor_map, skipped


def _load_qwen3_5_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    reader = gguf.GGUFReader(gguf_file)
    tensor_map = {t.name: t for t in reader.tensors}
    module_map = {
        getattr(module, "prefix"): module
        for _, module in model.named_modules()
        if hasattr(module, "prefix") and getattr(module, "prefix", "")
    }

    missing_keys: list[str] = []
    mapped_quant_modules = 0
    layer_types = getattr(hf_config, "layer_types", None)
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0))

    def _is_linear_layer(layer_idx: int) -> bool:
        if isinstance(layer_types, list) and layer_idx < len(layer_types):
            return layer_types[layer_idx] == "linear_attention"
        return False

    # Semantic integrity check before loading any weight.
    required_global = ("token_embd.weight", "output_norm.weight", "output.weight")
    for tensor_name in required_global:
        if tensor_name not in tensor_map:
            missing_keys.append(tensor_name)

    for layer_idx in range(num_layers):
        per_layer_required = [
            f"blk.{layer_idx}.attn_norm.weight",
            f"blk.{layer_idx}.post_attention_norm.weight",
        ]
        if _is_linear_layer(layer_idx):
            per_layer_required.extend([
                f"blk.{layer_idx}.attn_qkv.weight",
                f"blk.{layer_idx}.attn_gate.weight",
                f"blk.{layer_idx}.ssm_a",
                f"blk.{layer_idx}.ssm_alpha.weight",
                f"blk.{layer_idx}.ssm_beta.weight",
                f"blk.{layer_idx}.ssm_conv1d.weight",
                f"blk.{layer_idx}.ssm_dt.bias",
                f"blk.{layer_idx}.ssm_norm.weight",
                f"blk.{layer_idx}.ssm_out.weight",
            ])
        else:
            has_fused = f"blk.{layer_idx}.attn_qkv.weight" in tensor_map
            has_split = all(
                name in tensor_map for name in (
                    f"blk.{layer_idx}.attn_q.weight",
                    f"blk.{layer_idx}.attn_k.weight",
                    f"blk.{layer_idx}.attn_v.weight",
                )
            )
            # Relax check for attn_output if attn_qkv is present
            if not has_fused and not has_split:
                missing_keys.append(f"blk.{layer_idx}.attn_qkv.weight|attn_q+attn_k+attn_v")
            
            # Special case for Mamba-mix Qwen3.5: attn_output might be missing in Linear layers
            if not _is_linear_layer(layer_idx) and f"blk.{layer_idx}.attn_output.weight" not in tensor_map:
                if not has_fused:
                    missing_keys.append(f"blk.{layer_idx}.attn_output.weight")

        for tensor_name in per_layer_required:
            if tensor_name not in tensor_map:
                missing_keys.append(tensor_name)

        if int(getattr(hf_config, "num_experts", 0)) > 0:
            for moe_name in (
                f"blk.{layer_idx}.ffn_gate_exps.weight",
                f"blk.{layer_idx}.ffn_up_exps.weight",
                f"blk.{layer_idx}.ffn_down_exps.weight",
                f"blk.{layer_idx}.ffn_gate_inp.weight",
            ):
                if moe_name not in tensor_map:
                    missing_keys.append(moe_name)
        else:
            dense_pair = (
                f"blk.{layer_idx}.ffn_gate.weight",
                f"blk.{layer_idx}.ffn_up.weight",
            )
            if not all(name in tensor_map for name in dense_pair):
                missing_keys.append(f"blk.{layer_idx}.ffn_gate.weight+ffn_up.weight")
            if f"blk.{layer_idx}.ffn_down.weight" not in tensor_map:
                missing_keys.append(f"blk.{layer_idx}.ffn_down.weight")

    if missing_keys:
        preview = ", ".join(missing_keys[:16])
        if len(missing_keys) > 16:
            preview += f", ... (+{len(missing_keys) - 16} more)"
        raise RuntimeError(
            "Qwen3.5 GGUF semantic integrity check failed. Missing required tensors: "
            f"{preview}"
        )

    aggressive_qwen9 = _is_qwen9_aggressive_enabled(hf_config)
    prefer_fp32_linear = os.environ.get("FASTINFERENCE_QWEN35_FP32_LINEAR", "0") == "1"
    if not prefer_fp32_linear and int(getattr(hf_config, "num_experts", 0)) > 0:
        # For large MoE Qwen3.5, fp32 linear compute is often required for stability.
        prefer_fp32_linear = True
    preferred_linear_dtype = torch.float16 if aggressive_qwen9 else _resolve_hf_dtype(hf_config)

    for gguf_name, param in (
        ("token_embd.weight", model.model.embed_tokens.weight),
        ("output_norm.weight", model.model.norm.weight),
        ("output.weight", model.lm_head.weight),
    ):
        if gguf_name not in tensor_map:
            missing_keys.append(gguf_name)
            continue
        _load_param_from_gguf_tensor(param, tensor_map[gguf_name])

    for layer_idx, layer in enumerate(model.model.layers):
        attn_norm_name = f"blk.{layer_idx}.attn_norm.weight"
        post_norm_name = f"blk.{layer_idx}.post_attention_norm.weight"
        gate_name = f"blk.{layer_idx}.ffn_gate_inp.weight"
        q_norm_name = f"blk.{layer_idx}.attn_q_norm.weight"
        k_norm_name = f"blk.{layer_idx}.attn_k_norm.weight"
        ssm_a_name = f"blk.{layer_idx}.ssm_a"
        ssm_dt_name = f"blk.{layer_idx}.ssm_dt.bias"
        ssm_norm_name = f"blk.{layer_idx}.ssm_norm.weight"
        ssm_conv_name = f"blk.{layer_idx}.ssm_conv1d.weight"
        if attn_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.input_layernorm.weight, tensor_map[attn_norm_name])
        else:
            missing_keys.append(attn_norm_name)
        if post_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.post_attention_layernorm.weight, tensor_map[post_norm_name])
        else:
            missing_keys.append(post_norm_name)
        if hasattr(layer.mlp, "gate") and gate_name in tensor_map:
            _load_param_from_gguf_tensor(layer.mlp.gate.weight, tensor_map[gate_name])
        if hasattr(layer.self_attn, "q_norm") and q_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.self_attn.q_norm.weight, tensor_map[q_norm_name])
        if hasattr(layer.self_attn, "k_norm") and k_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.self_attn.k_norm.weight, tensor_map[k_norm_name])
        if hasattr(layer.self_attn, "ssm_state_a") and ssm_a_name in tensor_map:
            _load_param_from_gguf_tensor(layer.self_attn.ssm_state_a, tensor_map[ssm_a_name])
        if hasattr(layer.self_attn, "ssm_dt_bias") and ssm_dt_name in tensor_map:
            _load_param_from_gguf_tensor(layer.self_attn.ssm_dt_bias, tensor_map[ssm_dt_name])
        if hasattr(layer.self_attn, "ssm_norm_weight") and ssm_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.self_attn.ssm_norm_weight, tensor_map[ssm_norm_name])
        if hasattr(layer.self_attn, "ssm_conv1d_weight") and ssm_conv_name in tensor_map:
            conv_weight = _dequantize_gguf_tensor(
                tensor_map[ssm_conv_name],
                str(layer.self_attn.ssm_conv1d_weight.device),
                layer.self_attn.ssm_conv1d_weight.dtype,
            )
            if conv_weight.shape != layer.self_attn.ssm_conv1d_weight.shape:
                layer.self_attn.ssm_conv1d_weight = nn.Parameter(
                    conv_weight.contiguous(), requires_grad=False
                )
            else:
                layer.self_attn.ssm_conv1d_weight.data.copy_(conv_weight)

    prefix_aliases = {"attn_output": "attn_gate"}
    for prefix, module in module_map.items():
        if not hasattr(module, "quant_config") or module.quant_config is None:
            continue
        exact_name = f"{prefix}.weight"
        alias_name = None
        for old_suffix, new_suffix in prefix_aliases.items():
            if prefix.endswith(old_suffix):
                alias_name = f"{prefix[: -len(old_suffix)]}{new_suffix}.weight"
                break
        gguf_name = exact_name if exact_name in tensor_map else alias_name
        if gguf_name is None or gguf_name not in tensor_map:
            if exact_name.endswith(".attn_qkv.weight"):
                split_q = exact_name.replace(".attn_qkv.weight", ".attn_q.weight")
                split_k = exact_name.replace(".attn_qkv.weight", ".attn_k.weight")
                split_v = exact_name.replace(".attn_qkv.weight", ".attn_v.weight")
                if split_q in tensor_map and split_k in tensor_map and split_v in tensor_map:
                    q_dense = _dequantize_gguf_tensor(split_q := tensor_map[split_q], "cuda", torch.float16)
                    k_dense = _dequantize_gguf_tensor(split_k := tensor_map[split_k], "cuda", torch.float16)
                    v_dense = _dequantize_gguf_tensor(split_v := tensor_map[split_v], "cuda", torch.float16)
                    fused_qkv = torch.cat([q_dense, k_dense, v_dense], dim=0).contiguous()
                    module.quant_config = None
                    module.qweight = None
                    module.gguf_quant_type = None
                    module.gguf_shape = None
                    module.weight = nn.Parameter(fused_qkv, requires_grad=False)
                    module.input_size = fused_qkv.shape[1]
                    module.output_size = fused_qkv.shape[0]
                    if getattr(module, "bias", None) is not None:
                        module.bias = nn.Parameter(
                            torch.zeros(fused_qkv.shape[0], device=fused_qkv.device, dtype=fused_qkv.dtype),
                            requires_grad=False,
                        )
                    mapped_quant_modules += 1
                    continue
            if exact_name.endswith(".ffn_gate_up.weight"):
                gate_name = exact_name.replace(".ffn_gate_up.weight", ".ffn_gate.weight")
                up_name = exact_name.replace(".ffn_gate_up.weight", ".ffn_up.weight")
                if gate_name in tensor_map and up_name in tensor_map:
                    gate_dense = _dequantize_gguf_tensor(tensor_map[gate_name], "cuda", torch.float16)
                    up_dense = _dequantize_gguf_tensor(tensor_map[up_name], "cuda", torch.float16)
                    fused_gate_up = torch.cat([gate_dense, up_dense], dim=0).contiguous()
                    module.quant_config = None
                    module.qweight = None
                    module.gguf_quant_type = None
                    module.gguf_shape = None
                    module.weight = nn.Parameter(fused_gate_up, requires_grad=False)
                    module.input_size = fused_gate_up.shape[1]
                    module.output_size = fused_gate_up.shape[0]
                    if getattr(module, "bias", None) is not None:
                        module.bias = nn.Parameter(
                            torch.zeros(fused_gate_up.shape[0], device=fused_gate_up.device, dtype=fused_gate_up.dtype),
                            requires_grad=False,
                        )
                    mapped_quant_modules += 1
                    continue
            missing_keys.append(exact_name)
            continue
        gguf_tensor = tensor_map[gguf_name]
        
        # Optimization: Pre-dequantize Dense layers for FP8 acceleration
        # We skip SSM layers for now to be safe, only target standard Attn/MLP
        # For very large models (>20B parameters), we skip pre-dequantization to save VRAM.
        fp8_enabled = os.environ.get("FASTINFERENCE_DEEPSEEK_FP8", "0") == "1"
        num_params_est = sum(p.numel() for p in model.parameters())
        is_very_large = num_params_est > 20 * 10**9
        
        is_dense = ".attn_" in prefix or ".ffn_gate" in prefix or ".ffn_up" in prefix or ".ffn_down" in prefix
        if fp8_enabled and is_dense and ".ssm_" not in prefix and not is_very_large:
            w_fp16 = _dequantize_gguf_tensor(gguf_tensor, "cuda", torch.float16)
            module.weight = nn.Parameter(w_fp16, requires_grad=False)
            module.quant_config = None
            module.qweight = None
            module.output_size = w_fp16.shape[0]
            module.input_size = w_fp16.shape[1]
            mapped_quant_modules += 1
            continue

        module.qweight = nn.Parameter(_to_torch_tensor(gguf_tensor), requires_grad=False)
        module.gguf_quant_type = int(gguf_tensor.tensor_type)
        module.gguf_shape = tuple(int(x) for x in gguf_tensor.shape)
        if prefer_fp32_linear and any(
            token in prefix
            for token in (".attn_qkv", ".attn_gate", ".ssm_alpha", ".ssm_beta", ".ssm_out")
        ):
            module.gguf_compute_dtype = torch.float32
            module.gguf_output_dtype = preferred_linear_dtype
        else:
            module.gguf_compute_dtype = (
                torch.bfloat16 if preferred_linear_dtype == torch.bfloat16 else torch.float16
            )
            module.gguf_output_dtype = module.gguf_compute_dtype
        if len(module.gguf_shape) >= 2:
            module.input_size = int(module.gguf_shape[0])
            module.output_size = int(module.qweight.shape[0])
        if getattr(module, "bias", None) is not None:
            if module.bias.shape[0] != module.qweight.shape[0]:
                module.bias = nn.Parameter(
                    torch.zeros(module.qweight.shape[0], device=module.qweight.device, dtype=torch.float16),
                    requires_grad=False,
                )
            else:
                module.bias.data.zero_()
        mapped_quant_modules += 1

    # If no modules mapped, it might be because they are all very large or all were skipped.
    # We only raise if there are clearly missing keys that should have been there.
    # if mapped_quant_modules == 0:
    #    raise RuntimeError("No quantized GGUF tensors were mapped to LiteLinear modules.")

    critical_missing = [name for name in missing_keys if ".weight" in name and ".ffn_" not in name and ".ssm_" not in name]
    if critical_missing:
        preview = ", ".join(critical_missing[:12])
        if len(critical_missing) > 12:
            preview += f", ... (+{len(critical_missing) - 12} more)"
        raise RuntimeError(f"Critical GGUF tensors are missing for Qwen3.5 path: {preview}")


def _load_deepseek_v2_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    tensor_map, _ = _read_gguf_tensor_map_tolerant(gguf_file)
    module_map = {
        getattr(module, "prefix"): module
        for _, module in model.named_modules()
        if hasattr(module, "prefix") and getattr(module, "prefix", "")
    }

    missing_keys: list[str] = []
    mapped_quant_modules = 0
    preferred_dtype = _resolve_hf_dtype(hf_config)

    # Semantic integrity check before loading any weight.
    required_global = ("token_embd.weight", "output_norm.weight", "output.weight")
    for tensor_name in required_global:
        if tensor_name not in tensor_map:
            missing_keys.append(tensor_name)

    # Tolerant check for Attention: DeepSeek-V2/V3 use attn_q, 
    # but GLM-4.7-Flash (also arch=deepseek2) uses a more split structure.
    is_glm_flash = "blk.0.attn_q_a.weight" in tensor_map

    for layer_idx, layer in enumerate(model.model.layers):
        attn_norm_name = f"blk.{layer_idx}.attn_norm.weight"
        ffn_norm_name = f"blk.{layer_idx}.ffn_norm.weight"
        gate_name = f"blk.{layer_idx}.ffn_gate_inp.weight"
        if attn_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.input_layernorm.weight, tensor_map[attn_norm_name])
        else:
            missing_keys.append(attn_norm_name)
        if ffn_norm_name in tensor_map:
            _load_param_from_gguf_tensor(layer.post_attention_layernorm.weight, tensor_map[ffn_norm_name])
        else:
            missing_keys.append(ffn_norm_name)
        if hasattr(layer.mlp, "gate") and gate_name in tensor_map:
            _load_param_from_gguf_tensor(layer.mlp.gate.weight, tensor_map[gate_name])

        if is_glm_flash:
            # GLM-4.7-Flash specific fusion logic inside deepseek2 path
            q_a = f"blk.{layer_idx}.attn_q_a.weight"
            q_b = f"blk.{layer_idx}.attn_q_b.weight"
            kv_a = f"blk.{layer_idx}.attn_kv_a_mqa.weight"
            k_b = f"blk.{layer_idx}.attn_k_b.weight"
            v_b = f"blk.{layer_idx}.attn_v_b.weight"
            
            kv_rank = getattr(hf_config, "kv_lora_rank", 512)
            
            if all(x in tensor_map for x in (q_a, q_b)):
                wa = _dequantize_gguf_tensor(tensor_map[q_a], "cuda", torch.float16)
                wb = _dequantize_gguf_tensor(tensor_map[q_b], "cuda", torch.float16)
                if wa.shape[0] == layer.self_attn.q_proj.input_size: wa = wa.T
                layer.self_attn.q_proj.weight = nn.Parameter((wb @ wa).contiguous(), requires_grad=False)
                layer.self_attn.q_proj.quant_config = None
                
            if all(x in tensor_map for x in (kv_a, k_b, v_b)):
                kva = _dequantize_gguf_tensor(tensor_map[kv_a], "cuda", torch.float16)
                kb = _dequantize_gguf_tensor(tensor_map[k_b], "cuda", torch.float16)
                vb = _dequantize_gguf_tensor(tensor_map[v_b], "cuda", torch.float16)
                
                if kva.shape[0] < kva.shape[1]: kva = kva.T
                kva_main = kva[:, :kv_rank].T # [rank, in]
                
                # GLM-4.7 Flash special: kb is [nope_dim, heads * rank]
                # nope_dim is 192, heads is 20, rank is 512.
                # 20 * 512 = 10240. Matches!
                kb = kb.view(192, 20, 512)
                # vb is [256, 10240] -> [256, 20, 512]
                vb = vb.view(-1, 20, 512) 
                
                # We need to project kva_main [512, in] using kb [192, 20, 512]
                # Output should be [20, 192, in]
                k_nope = torch.einsum('r i, d h r -> h d i', kva_main, kb).reshape(-1, kva.shape[0])
                v_full = torch.einsum('r i, d h r -> h d i', kva_main, vb).reshape(-1, kva.shape[0])
                
                kva_rope = kva[:, kv_rank:].T # [64, in]
                
                fused_kv = torch.cat([k_nope, v_full, kva_rope], dim=0).contiguous()
                layer.self_attn.kv_proj.weight = nn.Parameter(fused_kv, requires_grad=False)
                layer.self_attn.kv_proj.quant_config = None

    # Pre-dequantize Dense layers for FP8 acceleration if enabled
    fp8_enabled = os.environ.get("FASTINFERENCE_DEEPSEEK_FP8", "0") == "1"
    
    # Shared Logic for all quantized layers (MoE + Dense)
    mapped_quant_modules = 0
    for prefix, module in module_map.items():
        if hasattr(module, "quant_config") and module.quant_config is not None:
            # Skip if already fused above
            if is_glm_flash and (".attn_q" in prefix or ".attn_kv" in prefix):
                continue
            
            exact_name = f"{prefix}.weight"
            if exact_name in tensor_map:
                gguf_tensor = tensor_map[exact_name]
                
                # Performance Optimization: For Dense layers (Attention/Non-MoE),
                # pre-dequantize to FP16 to enable FP8 LiteLinear path.
                is_moe_expert = ".ffn_gate_exps" in prefix or ".ffn_up_exps" in prefix or ".ffn_down_exps" in prefix
                
                if fp8_enabled and not is_moe_expert and (".attn_" in prefix or ".ffn_gate" in prefix or ".ffn_up" in prefix or ".ffn_down" in prefix):
                    # print(f"[DeepSeek-V2] Pre-dequantizing dense layer: {prefix}")
                    w_fp16 = _dequantize_gguf_tensor(gguf_tensor, "cuda", torch.float16)
                    module.weight = nn.Parameter(w_fp16, requires_grad=False)
                    module.quant_config = None
                    module.qweight = None
                    module.output_size = w_fp16.shape[0]
                    module.input_size = w_fp16.shape[1]
                    mapped_quant_modules += 1
                    continue

                # Pass crucial GGUF metadata for on-the-fly dequantization
                module.gguf_quant_type = int(gguf_tensor.tensor_type)
                # For MoE, gguf_shape is [input_size, inter_size]
                if len(gguf_tensor.shape) == 3:
                    # GLM-4.7 Flash: [2048, 1536, 64] -> (2048, 1536)
                    module.gguf_shape = (gguf_tensor.shape[0], gguf_tensor.shape[1])
                else:
                    module.gguf_shape = (gguf_tensor.shape[0], gguf_tensor.shape[1])
                
                module.load_weights([(exact_name, _to_torch_tensor(gguf_tensor))])
                
                module.gguf_compute_dtype = torch.float16
                module.gguf_output_dtype = torch.float16
                if len(module.gguf_shape) >= 2:
                    module.input_size = int(module.gguf_shape[0])
                    # Ensure output_size is correct for the weight
                    if hasattr(module, "qweight") and module.qweight is not None:
                        module.output_size = int(module.qweight.shape[0])
                
                if getattr(module, "bias", None) is not None:
                    if hasattr(module, "qweight") and module.qweight is not None:
                        if module.bias.shape[0] != module.qweight.shape[0]:
                            module.bias = nn.Parameter(
                                torch.zeros(module.qweight.shape[0], device=module.qweight.device, dtype=torch.float16),
                                requires_grad=False,
                            )
                        else:
                            module.bias.data.zero_()
                mapped_quant_modules += 1
            else:
                missing_keys.append(exact_name)

    if mapped_quant_modules == 0:
        raise RuntimeError("No quantized GGUF tensors were mapped for DeepSeekV2 path.")

    critical_missing = [name for name in missing_keys if ".weight" in name and ".ffn_" not in name]
    if critical_missing:
        preview = ", ".join(critical_missing[:12])
        if len(critical_missing) > 12:
            preview += f", ... (+{len(critical_missing) - 12} more)"
        raise RuntimeError(f"Critical GGUF tensors are missing for DeepSeekV2 path: {preview}")


def _load_glm_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    tensor_map, skipped_tensors = _read_gguf_tensor_map_tolerant(gguf_file)
    skipped_set = set(skipped_tensors)
    if skipped_tensors:
        # Keep loading attempt deterministic and fail-fast if critical tensors
        # are later missing; this warning helps explain parser limitations.
        print(f"[GLM GGUF] skipped unreadable tensors: {len(skipped_tensors)}")
        skipped_layers = sorted(
            {
                int(name.split(".")[1])
                for name in skipped_tensors
                if name.startswith("blk.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
            }
        )
        if skipped_layers:
            min_skipped = skipped_layers[0]
            num_layers = int(getattr(hf_config, "num_hidden_layers", 0))
            tail_ratio = sum(1 for x in skipped_layers if x >= min_skipped) / max(1, num_layers - min_skipped)
            if min_skipped >= 1 and tail_ratio > 0.8:
                raise RuntimeError(
                    f"GLM GGUF appears truncated/corrupted from layer {min_skipped} onward "
                    f"(unreadable tensors: {len(skipped_tensors)}). "
                    "Please re-export or re-download the GGUF model file."
                )

    missing_keys: list[str] = []
    for gguf_name, param in (
        ("token_embd.weight", model.embed_tokens.weight),
        ("output_norm.weight", model.norm.weight),
        ("output.weight", model.lm_head.weight),
    ):
        if gguf_name not in tensor_map:
            missing_keys.append(gguf_name)
            continue
        _load_param_from_gguf_tensor(param, tensor_map[gguf_name])

    num_layers = int(getattr(hf_config, "num_hidden_layers", 0))
    first_k_dense = int(getattr(hf_config, "first_k_dense_replace", 0))
    for i, layer in enumerate(model.layers):
        attn_norm = f"blk.{i}.attn_norm.weight"
        ffn_norm = f"blk.{i}.ffn_norm.weight"
        if attn_norm in tensor_map:
            _load_param_from_gguf_tensor(layer.input_layernorm.weight, tensor_map[attn_norm])
        else:
            missing_keys.append(attn_norm)
        if ffn_norm in tensor_map:
            _load_param_from_gguf_tensor(layer.post_attention_layernorm.weight, tensor_map[ffn_norm])
        else:
            missing_keys.append(ffn_norm)
        if i >= first_k_dense and hasattr(layer.mlp, "gate"):
            gate_name = f"blk.{i}.ffn_gate_inp.weight"
            if gate_name in tensor_map:
                _load_param_from_gguf_tensor(layer.mlp.gate.weight, tensor_map[gate_name])
            else:
                missing_keys.append(gate_name)

        # Fuse q_a/q_b into q_proj to match current Lite GLM path.
        q_a_name = f"blk.{i}.attn_q_a.weight"
        q_b_name = f"blk.{i}.attn_q_b.weight"
        if q_a_name in tensor_map and q_b_name in tensor_map:
            q_a = _dequantize_gguf_tensor(tensor_map[q_a_name], "cuda", torch.float16)
            q_b = _dequantize_gguf_tensor(tensor_map[q_b_name], "cuda", torch.float16)
            layer.self_attn.q_proj.quant_config = None
            layer.self_attn.q_proj.qweight = None
            layer.self_attn.q_proj.weight = nn.Parameter((q_b @ q_a).contiguous(), requires_grad=False)
            layer.self_attn.q_proj.output_size = layer.self_attn.q_proj.weight.shape[0]
        else:
            missing_keys.extend([name for name in (q_a_name, q_b_name) if name not in tensor_map])

        # Fuse kv_a/k_b/v_b into kv_proj approximate mapping.
        kv_a_name = f"blk.{i}.attn_kv_a_mqa.weight"
        k_b_name = f"blk.{i}.attn_k_b.weight"
        v_b_name = f"blk.{i}.attn_v_b.weight"
        if kv_a_name in tensor_map and k_b_name in tensor_map and v_b_name in tensor_map:
            kv_a = _dequantize_gguf_tensor(tensor_map[kv_a_name], "cuda", torch.float16)
            k_b = _dequantize_gguf_tensor(tensor_map[k_b_name], "cuda", torch.float16)
            v_b = _dequantize_gguf_tensor(tensor_map[v_b_name], "cuda", torch.float16)
            kv_lora_rank = int(getattr(hf_config, "kv_lora_rank", 0))
            kv_a_main = kv_a[:kv_lora_rank, :]
            kv_a_rope = kv_a[kv_lora_rank:, :]
            # k_b: [num_heads, kv_rank, qk_nope] -> [num_heads, qk_nope, kv_rank]
            k_nope = torch.matmul(k_b.transpose(1, 2), kv_a_main).contiguous()
            # v_b: [num_heads, v_head, kv_rank]
            v_full = torch.matmul(v_b, kv_a_main).contiguous()
            k_nope_flat = k_nope.view(-1, k_nope.shape[-1])
            v_full_flat = v_full.view(-1, v_full.shape[-1])
            kv_fused = torch.cat([k_nope_flat, v_full_flat, kv_a_rope], dim=0).contiguous()
            layer.self_attn.kv_proj.quant_config = None
            layer.self_attn.kv_proj.qweight = None
            layer.self_attn.kv_proj.weight = nn.Parameter(kv_fused, requires_grad=False)
            layer.self_attn.kv_proj.output_size = kv_fused.shape[0]
        else:
            missing_keys.extend([name for name in (kv_a_name, k_b_name, v_b_name) if name not in tensor_map])

        o_name = f"blk.{i}.attn_output.weight"
        if o_name in tensor_map:
            o_w = _dequantize_gguf_tensor(tensor_map[o_name], "cuda", torch.float16).contiguous()
            layer.self_attn.o_proj.quant_config = None
            layer.self_attn.o_proj.qweight = None
            layer.self_attn.o_proj.weight = nn.Parameter(o_w, requires_grad=False)
            layer.self_attn.o_proj.output_size = o_w.shape[0]
        else:
            missing_keys.append(o_name)

    module_map = {
        getattr(module, "prefix"): module
        for _, module in model.named_modules()
        if hasattr(module, "prefix") and getattr(module, "prefix", "")
    }
    for prefix, module in module_map.items():
        if not hasattr(module, "quant_config") or module.quant_config is None:
            continue
        exact_name = f"{prefix}.weight"
        alias_name = exact_name.replace(".mlp.", ".")
        gguf_name = exact_name if exact_name in tensor_map else alias_name
        if gguf_name not in tensor_map:
            if gguf_name not in skipped_set:
                missing_keys.append(exact_name)
            continue
        gguf_tensor = tensor_map[gguf_name]
        module.qweight = nn.Parameter(_to_torch_tensor(gguf_tensor), requires_grad=False)
        module.gguf_quant_type = int(gguf_tensor.tensor_type)
        module.gguf_shape = tuple(int(x) for x in gguf_tensor.shape)
        module.gguf_compute_dtype = torch.float16
        module.gguf_output_dtype = torch.float16
        if len(module.gguf_shape) >= 2:
            module.input_size = int(module.gguf_shape[0])
            module.output_size = int(module.qweight.shape[0])
        if getattr(module, "bias", None) is not None:
            if module.bias.shape[0] != module.qweight.shape[0]:
                module.bias = nn.Parameter(
                    torch.zeros(module.qweight.shape[0], device=module.qweight.device, dtype=torch.float16),
                    requires_grad=False,
                )
            else:
                module.bias.data.zero_()

    unresolved_missing = [name for name in missing_keys if name not in skipped_set]
    critical_missing = sorted(set(name for name in unresolved_missing if name.startswith("blk.")))
    if critical_missing:
        preview = ", ".join(critical_missing[:24])
        if len(critical_missing) > 24:
            preview += f", ... (+{len(critical_missing) - 24} more)"
        raise RuntimeError(f"Critical GGUF tensors are missing for GLM path: {preview}")

    if num_layers == 0:
        raise RuntimeError("Invalid GLM config: num_hidden_layers is 0.")


def get_model(vllm_config: VllmConfig) -> nn.Module:
    """LitevLLM model loader with safetensors and GGUF path."""
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", [])
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures, vllm_config.model_config)
    target_dtype = _resolve_target_dtype(vllm_config)
    if _is_qwen9_aggressive_enabled(vllm_config.model_config.hf_config) or \
       _is_deepseek_aggressive_enabled(vllm_config.model_config.hf_config):
        target_dtype = torch.float16
    model = model_cls(vllm_config).cuda().to(dtype=target_dtype)

    model_path = vllm_config.model_config.model
    if any(f.endswith(".safetensors") for f in os.listdir(model_path) if os.path.isdir(model_path)):
        from vllm.model_executor.model_loader.safetensors import load_safetensors_weights

        load_safetensors_weights(model, model_path)
    elif any(f.endswith(".gguf") for f in os.listdir(model_path) if os.path.isdir(model_path)):
        gguf_files = sorted(
            [os.path.join(model_path, filename) for filename in os.listdir(model_path) if filename.endswith(".gguf")]
        )
        if not gguf_files:
            raise RuntimeError("GGUF directory is detected but no .gguf file is found.")
        if any(("Qwen3_5" in arch) or ("Qwen2" in arch) for arch in architectures):
            _load_qwen3_5_gguf_weights(model, gguf_files[0], vllm_config.model_config.hf_config)
        elif any(("DeepseekV2" in arch) or ("DeepseekV3" in arch) for arch in architectures):
            _load_deepseek_v2_gguf_weights(model, gguf_files[0], vllm_config.model_config.hf_config)
        elif any(("Glm4MoeLite" in arch) or ("GlmForCausalLM" in arch) for arch in architectures):
            _load_glm_gguf_weights(model, gguf_files[0], vllm_config.model_config.hf_config)
        else:
            raise RuntimeError(f"GGUF loading is not implemented for architecture list: {architectures}")

    return model


def get_tokenizer(model_config: Any, **kwargs):
    """LitevLLM: Unified tokenizer loader via Registry."""
    from vllm.tokenizers.registry import get_tokenizer as registry_get_tokenizer

    return registry_get_tokenizer(model_config, **kwargs)
