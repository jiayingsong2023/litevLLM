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
    rows = int(packed.shape[0])
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    cols = int((packed.shape[-1] // type_size) * block_size)
    return ggml_dequantize_fallback(packed, quant_type, rows, cols, dtype)


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
            if not has_fused and not has_split:
                missing_keys.append(f"blk.{layer_idx}.attn_qkv.weight|attn_q+attn_k+attn_v")
            per_layer_required.append(f"blk.{layer_idx}.attn_output.weight")

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

    if mapped_quant_modules == 0:
        raise RuntimeError("No quantized GGUF tensors were mapped to LiteLinear modules.")

    critical_missing = [name for name in missing_keys if ".weight" in name and ".ffn_" not in name]
    if critical_missing:
        preview = ", ".join(critical_missing[:12])
        if len(critical_missing) > 12:
            preview += f", ... (+{len(critical_missing) - 12} more)"
        raise RuntimeError(f"Critical GGUF tensors are missing for Qwen3.5 path: {preview}")


def get_model(vllm_config: VllmConfig) -> nn.Module:
    """LitevLLM model loader with safetensors and GGUF path."""
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", [])
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures, vllm_config.model_config)
    target_dtype = _resolve_target_dtype(vllm_config)
    if _is_qwen9_aggressive_enabled(vllm_config.model_config.hf_config):
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
        if any("Qwen3_5" in arch for arch in architectures):
            _load_qwen3_5_gguf_weights(model, gguf_files[0], vllm_config.model_config.hf_config)
        else:
            raise RuntimeError(f"GGUF loading is not implemented for architecture list: {architectures}")

    return model


def get_tokenizer(model_config: Any, **kwargs):
    """LitevLLM: Unified tokenizer loader via Registry."""
    from vllm.tokenizers.registry import get_tokenizer as registry_get_tokenizer

    return registry_get_tokenizer(model_config, **kwargs)
