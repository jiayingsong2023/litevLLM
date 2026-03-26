# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from collections import defaultdict

class LRUWeightCache:
    def __init__(self, max_size=256):
        self.cache: Dict[int, Any] = {}
        self.keys = []
        self.max_size = max_size
    def get(self, key: int) -> Optional[Any]:
        if key in self.cache:
            self.keys.remove(key); self.keys.append(key)
            return self.cache[key]
        return None
    def put(self, key: int, value: Any):
        if key in self.cache: return
        if len(self.keys) >= self.max_size:
            old_key = self.keys.pop(0); del self.cache[old_key]
        self.cache[key] = value; self.keys.append(key)
    def clear(self):
        self.cache.clear()
        self.keys.clear()

_GLOBAL_WEIGHT_CACHE = LRUWeightCache(max_size=512)
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_USE_FP8_AWQ = os.environ.get("FASTINFERENCE_AWQ_FP8", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
_USE_BLOCK_FP8_AWQ = os.environ.get("FASTINFERENCE_AWQ_BLOCK_FP8", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
_USE_HIGH_FIDELITY_ALL_AWQ = os.environ.get("FASTINFERENCE_AWQ_HIGH_FIDELITY_ALL", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
_USE_HIGH_FIDELITY_PREFIX_MATCH = os.environ.get(
    "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIX_MATCH", "0"
).strip().lower() in ("1", "true", "yes", "on")
_USE_AWQ_DENSE_FALLBACK_CACHE = os.environ.get(
    "FASTINFERENCE_AWQ_DENSE_FALLBACK_CACHE", "1"
).strip().lower() in ("1", "true", "yes", "on")


def _default_awq_dense_fallback_max_gb() -> float:
    # Large shared-memory iGPU setups can keep a substantially larger dense-fallback
    # cache, which reduces repeated dequant on decode.
    if not torch.cuda.is_available():
        return 4.0
    try:
        total_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    except Exception:
        return 4.0
    if total_gb >= 48.0:
        return 14.0
    if total_gb >= 32.0:
        return 8.0
    if total_gb >= 16.0:
        return 4.0
    return 2.0


_AWQ_DENSE_FALLBACK_MAX_BYTES = int(
    float(
        os.environ.get(
            "FASTINFERENCE_AWQ_DENSE_FALLBACK_MAX_GB",
            str(_default_awq_dense_fallback_max_gb()),
        )
    )
    * (1024**3)
)
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max) if _FP8_DTYPE is not None else None
_FP8_MIN_SCALE = 1.0 / (_FP8_MAX * 512.0) if _FP8_MAX is not None else None
_HIGH_FIDELITY_PREFIXES = tuple(
    part.strip()
    for part in os.environ.get(
        "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIXES",
        ",".join(
            [
                ".linear_attn.in_proj_qkv",
                ".linear_attn.in_proj_z",
                ".linear_attn.in_proj_a",
                ".linear_attn.in_proj_b",
                ".linear_attn.out_proj",
                ".self_attn.q_proj",
                ".self_attn.k_proj",
                ".self_attn.v_proj",
                ".self_attn.o_proj",
                ".shared_expert.gate_proj",
                ".shared_expert.up_proj",
                ".shared_expert.down_proj",
                ".mlp.gate_proj",
                ".mlp.up_proj",
                ".mlp.down_proj",
            ]
        ),
    ).split(",")
    if part.strip()
)

_AWQ_RUNTIME_STATS: Dict[str, int] = defaultdict(int)
_AWQ_DENSE_FALLBACK_BYTES = 0


def _awq_stat_inc(key: str, delta: int = 1) -> None:
    _AWQ_RUNTIME_STATS[key] += delta


def reset_awq_runtime_stats() -> None:
    global _AWQ_DENSE_FALLBACK_BYTES
    _AWQ_RUNTIME_STATS.clear()
    _AWQ_DENSE_FALLBACK_BYTES = 0


def get_awq_runtime_stats() -> Dict[str, int]:
    return dict(_AWQ_RUNTIME_STATS)


def clear_global_weight_cache() -> None:
    global _AWQ_DENSE_FALLBACK_BYTES
    _GLOBAL_WEIGHT_CACHE.clear()
    _AWQ_DENSE_FALLBACK_BYTES = 0


def _dtype_cache_key(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _try_get_cached_dense_fallback(
    cached_weight: dict[str, Any],
    x: torch.Tensor,
    builder_fn,
) -> torch.Tensor:
    global _AWQ_DENSE_FALLBACK_BYTES

    if not _USE_AWQ_DENSE_FALLBACK_CACHE:
        _awq_stat_inc("dense_fallback_cache_disabled")
        return builder_fn()

    if not bool(cached_weight.get("allow_dense_fallback_cache", True)):
        _awq_stat_inc("dense_fallback_cache_disallowed")
        return builder_fn()

    key = _dtype_cache_key(x.dtype)
    dense_cache = cached_weight.setdefault("dense_fallback_cache", {})
    cached_dense = dense_cache.get(key)
    if cached_dense is not None:
        _awq_stat_inc("dense_fallback_cache_hit")
        return cached_dense

    _awq_stat_inc("dense_fallback_cache_miss")
    dense_weight = builder_fn()
    dense_bytes = int(dense_weight.numel() * dense_weight.element_size())
    if (
        dense_bytes <= 0
        or _AWQ_DENSE_FALLBACK_MAX_BYTES <= 0
        or _AWQ_DENSE_FALLBACK_BYTES + dense_bytes > _AWQ_DENSE_FALLBACK_MAX_BYTES
    ):
        _awq_stat_inc("dense_fallback_cache_skip_capacity")
        return dense_weight

    dense_cache[key] = dense_weight
    _AWQ_DENSE_FALLBACK_BYTES += dense_bytes
    _awq_stat_inc("dense_fallback_cache_store")
    return dense_weight


def _match_weight_dtype(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if weight.dtype == x.dtype:
        return weight
    return weight.to(dtype=x.dtype)


def _can_use_fp8_weight_cache(x: torch.Tensor) -> bool:
    return bool(_USE_FP8_AWQ and _FP8_DTYPE is not None and x.device.type != "cpu")


def _should_use_high_fidelity_awq(prefix: str, force_high_fidelity: bool = False) -> bool:
    if force_high_fidelity or _USE_HIGH_FIDELITY_ALL_AWQ:
        return True
    if not _USE_HIGH_FIDELITY_PREFIX_MATCH:
        return False
    if not prefix:
        return False
    return any(token in prefix for token in _HIGH_FIDELITY_PREFIXES)


def _quantize_per_tensor_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_f32 = x.to(torch.float32)
    absmax = x_f32.abs().amax()
    scale = (absmax / _FP8_MAX).clamp(min=_FP8_MIN_SCALE)
    x_fp8 = (x_f32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return x_fp8.contiguous(), scale.reshape(1).to(dtype=torch.float32, device=x.device)


def _build_fp8_weight_cache(dense_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    if _USE_BLOCK_FP8_AWQ:
        try:
            from vllm.model_executor.moe_fp8_utils import (
                dims_ok_for_moe_fp8,
                fp8_block_quantize_2d,
            )

            if dims_ok_for_moe_fp8(dense_weight.shape[1], dense_weight.shape[0]):
                weight_fp8, weight_scale = fp8_block_quantize_2d(dense_weight.contiguous())
                return {
                    "mode": "block_fp8",
                    "weight": weight_fp8,
                    "scale": weight_scale,
                }
        except Exception:
            pass

    weight_fp8, weight_scale = _quantize_per_tensor_fp8(dense_weight)
    return {
        "mode": "fp8",
        "weight": weight_fp8,
        "scale": weight_scale,
    }


def _scaled_mm_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _awq_stat_inc("fp8_scaled_mm_attempt")
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_fp8, x_scale = _quantize_per_tensor_fp8(x_2d)
    output_shape = [x_2d.shape[0], weight_fp8.shape[0]]
    bias_arg = bias.to(dtype=x.dtype) if bias is not None else None
    b_mat = weight_fp8.t().contiguous()

    try:
        output = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            x_fp8,
            b_mat,
            x.dtype,
            x_scale,
            weight_scale,
            bias_arg,
        )
        _awq_stat_inc("fp8_scaled_mm_rocm_kernel_ok")
    except Exception:
        _awq_stat_inc("fp8_scaled_mm_rocm_kernel_fail")
        output = torch._scaled_mm(
            x_fp8,
            b_mat,
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias_arg,
        )
        _awq_stat_inc("fp8_scaled_mm_torch_ok")
        if isinstance(output, tuple):
            output = output[0]

    return output.view(*x.shape[:-1], weight_fp8.shape[0])


def _linear_from_fp8_cache(
    x: torch.Tensor,
    cached_weight: dict[str, Any],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Fallback for platforms without FP8 GEMM support: keep the persistent cache in FP8,
    # but only materialize the current layer's dense weight for this single matmul.
    _awq_stat_inc("fp8_cache_to_dense_fallback")
    dense_weight = _try_get_cached_dense_fallback(
        cached_weight,
        x,
        lambda: cached_weight["weight"].to(dtype=x.dtype),
    )
    return torch.nn.functional.linear(x, dense_weight, bias)


def _linear_from_block_fp8_cache(
    x: torch.Tensor,
    cached_weight: dict[str, Any],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _awq_stat_inc("block_fp8_cache_to_dense_fallback")
    from vllm.model_executor.moe_fp8_utils import moe_fp8_dequant_to_linear_weight

    dense_weight = _try_get_cached_dense_fallback(
        cached_weight,
        x,
        lambda: moe_fp8_dequant_to_linear_weight(
            cached_weight["weight"],
            cached_weight["scale"],
            x.dtype,
        ),
    )
    return torch.nn.functional.linear(x, dense_weight, bias)


def _apply_linear_with_cached_weight(
    x: torch.Tensor,
    cached_weight: Any,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(cached_weight, dict) and cached_weight.get("mode") == "block_fp8":
        _awq_stat_inc("cached_mode_block_fp8")
        return _linear_from_block_fp8_cache(
            x,
            cached_weight,
            bias=bias,
        )
    if isinstance(cached_weight, dict) and cached_weight.get("mode") == "fp8":
        _awq_stat_inc("cached_mode_fp8")
        try:
            return _scaled_mm_linear(
                x,
                cached_weight["weight"],
                cached_weight["scale"],
                bias=bias,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "torch._scaled_mm is only supported" not in message:
                raise
            _awq_stat_inc("fp8_scaled_mm_runtime_unsupported")
            return _linear_from_fp8_cache(x, cached_weight, bias=bias)
        except AttributeError:
            _awq_stat_inc("fp8_scaled_mm_attribute_missing")
            return _linear_from_fp8_cache(x, cached_weight, bias=bias)
    _awq_stat_inc("cached_mode_dense")
    return torch.nn.functional.linear(x, _match_weight_dtype(cached_weight, x), bias)

def dequantize_q4k_pytorch(qweight: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    """Accurate Q4_K dequantization using gguf library reference implementation."""
    try:
        from gguf import dequantize, GGMLQuantizationType
        import numpy as np
        w_np = qweight.cpu().numpy()
        dequant_np = dequantize(w_np, GGMLQuantizationType.Q4_K)
        res = torch.from_numpy(np.array(dequant_np, copy=True)).to(device=qweight.device, dtype=torch.float16)
        # Reshape with safety — gguf.dequantize returns flat or 2D
        total = n_rows * n_cols
        if res.numel() >= total:
            return res.view(-1)[:total].view(n_rows, n_cols)
        else:
            out = torch.zeros(total, device=res.device, dtype=res.dtype)
            out[:res.numel()] = res.view(-1)
            return out.view(n_rows, n_cols)
    except Exception as e:
        raise RuntimeError(f"Q4_K Dequant Error: {e} ({qweight.shape}, R:{n_rows}, C:{n_cols})")

def dequantize_q6k_pytorch(qweight: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    """Accurate Q6_K dequantization using gguf library reference implementation."""
    try:
        from gguf import dequantize, GGMLQuantizationType
        import numpy as np
        w_np = qweight.cpu().numpy()
        dequant_np = dequantize(w_np, GGMLQuantizationType.Q6_K)
        res = torch.from_numpy(np.array(dequant_np, copy=True)).to(device=qweight.device, dtype=torch.float16)
        total = n_rows * n_cols
        if res.numel() >= total:
            return res.view(-1)[:total].view(n_rows, n_cols)
        else:
            out = torch.zeros(total, device=res.device, dtype=res.dtype)
            out[:res.numel()] = res.view(-1)
            return out.view(n_rows, n_cols)
    except Exception as e:
        raise RuntimeError(f"Q6_K Dequant Error: {e} ({qweight.shape}, R:{n_rows}, C:{n_cols})")

def dequantize_awq_pytorch(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    try:
        n_rows, n_cols_packed = qweight.shape; n_cols = n_cols_packed * 8
        shifts = torch.arange(0, 32, 4, device=qweight.device)
        qs = ((qweight.unsqueeze(-1) >> shifts) & 0x0F).view(n_rows, n_cols).to(torch.float32)
        zs = ((qzeros.unsqueeze(-1) >> shifts) & 0x0F).view(qzeros.shape[0], -1).to(torch.float32)
        n_groups = n_cols // group_size; qs = qs.view(n_rows, n_groups, group_size)
        res = (qs - zs.unsqueeze(-1)) * scales.to(torch.float32).unsqueeze(-1)
        return res.view(n_rows, n_cols).to(torch.float16)
    except Exception as e: raise RuntimeError(f"AWQ PyTorch Dequant Error: {e}")


def dequantize_symmetric_packed_int4_pytorch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    HF compressed-tensors / pack-quantized checkpoints often ship int4 weights with
    weight_packed + weight_scale only (no zero-point tensor). Nibbles are unsigned
    in [0, 15]; map to signed with (q - 8) * scale per group (same nibble layout as AWQ).
    """
    try:
        n_rows, n_cols_packed = qweight.shape
        n_cols = n_cols_packed * 8
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
        qs = ((qweight.unsqueeze(-1) >> shifts) & 0x0F).view(n_rows, n_cols).to(torch.float32)
        qs = qs - 8.0
        if n_cols % group_size != 0:
            raise RuntimeError(f"n_cols={n_cols} not divisible by group_size={group_size}")
        n_groups = n_cols // group_size
        qs = qs.view(n_rows, n_groups, group_size)
        res = qs * scales.to(torch.float32).unsqueeze(-1)
        return res.view(n_rows, n_cols)
    except Exception as e:
        raise RuntimeError(f"Symmetric packed int4 dequant error: {e}")


def dequantize_symmetric_packed_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    original_shape: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Prefer the same compressed-tensors unpack/dequant path the checkpoint expects.
    Fall back to the local PyTorch implementation when the dependency is unavailable.
    """
    qweight_int32 = qweight.to(torch.int32)
    unpack_shape = original_shape
    if unpack_shape is None:
        unpack_shape = (qweight.shape[0], qweight.shape[1] * 8)

    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
            unpack_from_int32,
        )
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        unpacked = unpack_from_int32(qweight_int32, 4, unpack_shape)
        dense_weight = dequantize(
            unpacked.to(torch.float32),
            scales.to(torch.float32),
            None,
            dtype=torch.float16,
        )
        return dense_weight
    except ImportError:
        dense_weight = dequantize_symmetric_packed_int4_pytorch(
            qweight_int32,
            scales,
            group_size=group_size,
        )
        if original_shape is not None:
            dense_weight = dense_weight[: original_shape[0], : original_shape[1]].contiguous()
        return dense_weight

class QuantizedLinearWeight(nn.Module, ABC):
    def __init__(self): super().__init__(); self.weight_id = id(self)
    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor: pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, quant_type=2, prefer_fused=True, original_shape=None, slice_offset=0):
        super().__init__(); self.qweight = nn.Parameter(qweight, requires_grad=False); self.scales = nn.Parameter(scales, requires_grad=False); self.quant_type = quant_type; self.prefer_fused = prefer_fused; self.original_shape = original_shape; self.slice_offset = slice_offset
    def matmul(self, x, bias=None):
        n_rows = self.qweight.shape[0]; n_cols = (self.qweight.shape[1] // 144 * 256) if self.quant_type >= 12 else (self.qweight.shape[1] // 18 * 32)
        bs = x.shape[0] if x.dim() > 1 else 1
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            if self.quant_type == 2: from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant; cached_w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
            elif self.quant_type == 12: cached_w = dequantize_q4k_pytorch(self.qweight, n_rows, n_cols)
            elif self.quant_type == 14: cached_w = dequantize_q6k_pytorch(self.qweight, n_rows, n_cols)
            if self.original_shape is not None or self.slice_offset > 0:
                os0 = self.original_shape[0] if self.original_shape else n_rows; os1 = self.original_shape[1] if self.original_shape else n_cols
                cached_w = cached_w[self.slice_offset : self.slice_offset + os0, :os1].contiguous()
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return _apply_linear_with_cached_weight(x, cached_w, bias)

class AWQWeight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, qzeros, group_size=128, prefix: str = "", high_fidelity: bool = False):
        super().__init__(); self.qweight = nn.Parameter(qweight, requires_grad=False); self.scales = nn.Parameter(scales, requires_grad=False); self.qzeros = nn.Parameter(qzeros, requires_grad=False); self.group_size = group_size; self.prefix = prefix; self.high_fidelity = high_fidelity
    def matmul(self, x, bias=None):
        if _should_use_high_fidelity_awq(self.prefix, self.high_fidelity):
            _awq_stat_inc("high_fidelity_awq_direct_dequant")
            try:
                from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
                dense_weight = awq_dequantize_triton(self.qweight, self.scales, self.qzeros, self.group_size)
            except:
                dense_weight = dequantize_awq_pytorch(self.qweight, self.scales, self.qzeros, self.group_size)
            return torch.nn.functional.linear(x, _match_weight_dtype(dense_weight, x), bias)
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            try:
                from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
                cached_w = awq_dequantize_triton(self.qweight, self.scales, self.qzeros, self.group_size)
            except: cached_w = dequantize_awq_pytorch(self.qweight, self.scales, self.qzeros, self.group_size)
            if _can_use_fp8_weight_cache(x):
                cached_w = _build_fp8_weight_cache(cached_w)
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return _apply_linear_with_cached_weight(x, cached_w, bias)


class PackedInt4Weight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, group_size=128, original_shape: Optional[tuple[int, int]] = None, prefix: str = "", high_fidelity: bool = False):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.group_size = group_size
        self.original_shape = original_shape
        self.prefix = prefix
        self.high_fidelity = high_fidelity

    def matmul(self, x, bias=None):
        if _should_use_high_fidelity_awq(self.prefix, self.high_fidelity):
            _awq_stat_inc("high_fidelity_packed_int4_direct_dequant")
            dense_weight = dequantize_symmetric_packed_int4(
                self.qweight,
                self.scales,
                group_size=self.group_size,
                original_shape=self.original_shape,
            )
            return torch.nn.functional.linear(x, _match_weight_dtype(dense_weight, x), bias)
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            cached_w = dequantize_symmetric_packed_int4(
                self.qweight,
                self.scales,
                group_size=self.group_size,
                original_shape=self.original_shape,
            )
            if _can_use_fp8_weight_cache(x):
                cached_w = _build_fp8_weight_cache(cached_w)
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return _apply_linear_with_cached_weight(x, cached_w, bias)
