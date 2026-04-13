# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict
from collections import defaultdict


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _tensor_nbytes(t: Any) -> int:
    if isinstance(t, torch.Tensor):
        return int(t.numel() * t.element_size())
    if isinstance(t, dict):
        return sum(_tensor_nbytes(v) for v in t.values())
    if isinstance(t, (tuple, list)):
        return sum(_tensor_nbytes(v) for v in t)
    return 0


@dataclass(frozen=True)
class AWQExecutionPolicy:
    prefer_fused: bool
    allow_dense_cache: bool
    cache_scope: str
    fused_scope: str

class LRUWeightCache:
    def __init__(self, max_size: int = 256, max_bytes: int = 0):
        self.cache: Dict[int, Any] = {}
        self.keys = []
        self.max_size = max_size
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.item_bytes: Dict[int, int] = {}

    def _evict_one(self) -> None:
        if not self.keys:
            return
        old_key = self.keys.pop(0)
        self.cache.pop(old_key, None)
        old_bytes = self.item_bytes.pop(old_key, 0)
        self.current_bytes = max(0, self.current_bytes - old_bytes)

    def get(self, key: int) -> Optional[Any]:
        if key in self.cache:
            self.keys.remove(key)
            self.keys.append(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: Any):
        if key in self.cache:
            return
        value_bytes = _tensor_nbytes(value)
        if self.max_bytes > 0 and value_bytes > self.max_bytes:
            return
        while self.max_size > 0 and len(self.keys) >= self.max_size:
            self._evict_one()
        while self.max_bytes > 0 and self.current_bytes + value_bytes > self.max_bytes:
            if not self.keys:
                break
            self._evict_one()
        if self.max_bytes > 0 and self.current_bytes + value_bytes > self.max_bytes:
            return
        self.cache[key] = value
        self.keys.append(key)
        self.item_bytes[key] = value_bytes
        self.current_bytes += value_bytes

    def get_memory_stats(self) -> Dict[str, int]:
        return {
            "cache_items": int(len(self.keys)),
            "cache_bytes": int(self.current_bytes),
            "cache_max_bytes": int(self.max_bytes),
            "cache_max_items": int(self.max_size),
        }

    def clear(self):
        self.cache.clear()
        self.keys.clear()
        self.item_bytes.clear()
        self.current_bytes = 0
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
_USE_AWQ_LEGACY_CACHE = _env_truthy("FASTINFERENCE_AWQ_LEGACY_CACHE", "0")
_GLOBAL_WEIGHT_CACHE = LRUWeightCache(
    max_size=512,
    max_bytes=0 if _USE_AWQ_LEGACY_CACHE else _AWQ_DENSE_FALLBACK_MAX_BYTES,
)
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


def _awq_stat_inc(key: str, delta: int = 1) -> None:
    _AWQ_RUNTIME_STATS[key] += delta


def _awq_stat_set(key: str, value: int) -> None:
    _AWQ_RUNTIME_STATS[key] = int(value)


def _env_awq_prefer_fused_default() -> bool:
    # Default on: Triton fused dequant+GEMM is the primary AWQ compute path.
    return _env_truthy("FASTINFERENCE_AWQ_FUSED_GEMM", "1")


def _env_awq_fused_gemm_force() -> bool:
    return _env_truthy("FASTINFERENCE_AWQ_FUSED_GEMM_FORCE", "0")


def _env_awq_matmul_cache_before_fused() -> bool:
    """
    When True and the global LRU already holds materialized weights (FP8/block or dense),
    run torch.nn.functional.linear (BLAS) instead of Triton fused.

    Default False: prefer the fused Triton AWQ GEMM path whenever policy allows it. Set
    FASTINFERENCE_AWQ_MATMUL_CACHE_BEFORE_FUSED=1 if you want BLAS when the cache is hot
    (e.g. steady-state throughput on some stacks). Disabled when
    FASTINFERENCE_AWQ_FUSED_GEMM_FORCE=1.
    """
    if _env_awq_fused_gemm_force():
        return False
    return _env_truthy("FASTINFERENCE_AWQ_MATMUL_CACHE_BEFORE_FUSED", "0")


def _env_awq_cache_scope() -> str:
    scope = os.environ.get("FASTINFERENCE_AWQ_CACHE_SCOPE", "all").strip().lower()
    if scope not in ("all", "attention_only", "off"):
        return "all"
    return scope


def _env_awq_fused_scope(profile_hint: str) -> str:
    matrix = os.environ.get("FASTINFERENCE_AWQ_POLICY_MATRIX", "balanced").strip().lower()
    if matrix not in ("safe", "balanced", "throughput", "strict"):
        matrix = "balanced"
    # Direct env override always wins.
    raw_override = os.environ.get("FASTINFERENCE_AWQ_FUSED_SCOPE")
    if raw_override is not None and raw_override.strip() != "":
        scope = raw_override.strip().lower()
        if scope in ("all", "attention_only", "off"):
            return scope
    # Profile-aware default matrix (no override):
    # - qwen35_9b_awq: safe=attention_only, balanced|throughput=all, strict=off
    if profile_hint == "qwen35_9b_awq":
        default_scope = (
            "all" if matrix in ("balanced", "throughput")
            else "off" if matrix == "strict"
            else "attention_only"
        )
    else:
        default_scope = "all" if matrix in ("balanced", "throughput") else "attention_only"
    scope = os.environ.get("FASTINFERENCE_AWQ_FUSED_SCOPE", default_scope).strip().lower()
    if scope not in ("all", "attention_only", "off"):
        return default_scope
    return scope


def _is_attention_like_prefix(prefix: str) -> bool:
    if not prefix:
        return False
    markers = (
        ".self_attn.",
        ".linear_attn.",
        ".attn.",
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".o_proj",
    )
    return any(m in prefix for m in markers)


def resolve_awq_execution_policy(
    prefix: str,
    x: torch.Tensor,
    profile_hint: str = "",
) -> AWQExecutionPolicy:
    cache_scope = _env_awq_cache_scope()
    fused_scope = _env_awq_fused_scope(profile_hint)
    allow_dense_cache = _USE_AWQ_DENSE_FALLBACK_CACHE
    if cache_scope == "off":
        allow_dense_cache = False
    elif cache_scope == "attention_only" and not _is_attention_like_prefix(prefix):
        allow_dense_cache = False
    return AWQExecutionPolicy(
        prefer_fused=_env_awq_prefer_fused_default(),
        allow_dense_cache=allow_dense_cache,
        cache_scope=cache_scope,
        fused_scope=fused_scope,
    )


def should_allow_dense_cache(prefix: str, policy: AWQExecutionPolicy) -> bool:
    if not policy.allow_dense_cache:
        return False
    if policy.cache_scope == "attention_only":
        return _is_attention_like_prefix(prefix)
    return True


def should_use_awq_fused_path(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    group_size: int,
    prefix: str,
    policy: AWQExecutionPolicy,
) -> tuple[bool, str]:
    if _env_awq_fused_gemm_force():
        # print(f">>>> DEBUG: Fused path FORCED for {prefix}")
        return True, "force_on"
    if not (policy.prefer_fused or _env_awq_fused_gemm_force()):
        return False, "fused_disabled"
    
    try:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_fused_capability_check,
        )
        res, reason = awq_fused_capability_check(
            x, qweight, scales, qzeros, group_size
        )
        return res, reason
    except Exception as exc:
        return False, f"fused_capability_check_error:{type(exc).__name__}"


def reset_awq_runtime_stats() -> None:
    _AWQ_RUNTIME_STATS.clear()
    _awq_stat_set("awq_matmul_calls", 0)
    _awq_stat_set("awq_fused_attempt", 0)
    _awq_stat_set("awq_fused_success", 0)
    _awq_stat_set("awq_cache_hits", 0)
    _awq_stat_set("awq_cache_misses", 0)
    _awq_stat_set("awq_dense_builds", 0)
    _awq_stat_set("awq_dense_cache_bytes_current", 0)


def get_awq_runtime_stats() -> Dict[str, int]:
    out = dict(_AWQ_RUNTIME_STATS)
    out.update(_GLOBAL_WEIGHT_CACHE.get_memory_stats())
    return out


def clear_global_weight_cache() -> None:
    _GLOBAL_WEIGHT_CACHE.clear()
    _awq_stat_set("awq_dense_cache_bytes_current", 0)


def _match_weight_dtype(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if weight.dtype == x.dtype:
        return weight
    return weight.to(dtype=x.dtype)


def _should_use_high_fidelity_awq(prefix: str, force_high_fidelity: bool = False) -> bool:
    if force_high_fidelity or _USE_HIGH_FIDELITY_ALL_AWQ:
        return True
    if not _USE_HIGH_FIDELITY_PREFIX_MATCH:
        return False
    if not prefix:
        return False
    return any(token in prefix for token in _HIGH_FIDELITY_PREFIXES)


def _apply_linear_with_cached_weight(
    x: torch.Tensor,
    cached_weight: Any,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(cached_weight, dict):
        raise RuntimeError(
            "FP8/block_fp8 weight cache entries were removed; restart the engine. "
            "Global LRU now stores dense tensors only (AWQ fallback / GGUF)."
        )
    _awq_stat_inc("cached_mode_dense")
    return torch.nn.functional.linear(x, _match_weight_dtype(cached_weight, x), bias)


def _awq_cache_put(weight_id: int, value: Any) -> None:
    _GLOBAL_WEIGHT_CACHE.put(weight_id, value)
    _awq_stat_set("awq_dense_cache_bytes_current", _GLOBAL_WEIGHT_CACHE.current_bytes)

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
            _awq_cache_put(self.weight_id, cached_w)
        return _apply_linear_with_cached_weight(x, cached_w, bias)

class AWQWeight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, qzeros, group_size=128, prefix: str = "", high_fidelity: bool = False, profile_hint: str = ""):
        super().__init__(); self.qweight = nn.Parameter(qweight, requires_grad=False); self.scales = nn.Parameter(scales, requires_grad=False); self.qzeros = nn.Parameter(qzeros, requires_grad=False); self.group_size = group_size; self.prefix = prefix; self.high_fidelity = high_fidelity; self.profile_hint = profile_hint
    def matmul(self, x, bias=None):
        _awq_stat_inc("awq_matmul_calls")
        
        # FAST PATH: Cached Decision
        if hasattr(self, "_cached_fused_decision"):
            if self._cached_fused_decision:
                try:
                    from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm_safe
                    out, used_fused, reason = awq_fused_gemm_safe(
                        x.reshape(-1, x.shape[-1]).contiguous(),
                        self.qweight, self.scales, self.qzeros, int(self.group_size), bias=bias,
                    )
                    if used_fused: return out.view(*x.shape[:-1], out.shape[-1])
                except: pass
            # Fallback to cached dense or dequant
            cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
            if cached_w is not None:
                # print(f">>>> DEBUG: Using CACHED DENSE for {self.prefix}")
                return _apply_linear_with_cached_weight(x, cached_w, bias)

        # SLOW PATH: First time resolution
        if _should_use_high_fidelity_awq(self.prefix, self.high_fidelity):
            # print(f">>>> DEBUG: High Fidelity Direct Dequant for {self.prefix}")
            self._cached_fused_decision = False
            return self._slow_matmul_dequant(x, bias)

        policy = resolve_awq_execution_policy(self.prefix, x, self.profile_hint)
        use_fused, _ = should_use_awq_fused_path(
            x=x, qweight=self.qweight, scales=self.scales, qzeros=self.qzeros,
            group_size=self.group_size, prefix=self.prefix, policy=policy,
        )
        self._cached_fused_decision = use_fused
        return self.matmul(x, bias) # Re-run with fast path

    def _slow_matmul_dequant(self, x, bias):
        try:
            from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
            dense_weight = awq_dequantize_triton(self.qweight, self.scales, self.qzeros, self.group_size)
        except:
            dense_weight = dequantize_awq_pytorch(self.qweight, self.scales, self.qzeros, self.group_size)
        return torch.nn.functional.linear(x, _match_weight_dtype(dense_weight, x), bias)


class PackedInt4Weight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, group_size=128, original_shape: Optional[tuple[int, int]] = None, prefix: str = "", high_fidelity: bool = False, profile_hint: str = ""):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.group_size = group_size
        self.original_shape = original_shape
        self.prefix = prefix
        self.high_fidelity = high_fidelity
        self.profile_hint = profile_hint

    def matmul(self, x, bias=None):
        _awq_stat_inc("awq_matmul_calls")

        def _dense_fallback() -> torch.Tensor:
            cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
            if cached_w is not None:
                return _apply_linear_with_cached_weight(x, cached_w, bias)
            # Use the local unpack/dequant path for stability with pack-quantized
            # checkpoints that can diverge under external helper backends.
            dense_weight = dequantize_symmetric_packed_int4_pytorch(
                self.qweight.to(torch.int32),
                self.scales,
                group_size=self.group_size,
            )
            if self.original_shape is not None:
                dense_weight = dense_weight[
                    : self.original_shape[0], : self.original_shape[1]
                ].contiguous()
            _awq_cache_put(self.weight_id, dense_weight)
            return torch.nn.functional.linear(x, _match_weight_dtype(dense_weight, x), bias)

        use_fused_cached = getattr(self, "_cached_fused_decision", None)
        if use_fused_cached is True:
            try:
                from vllm.kernels.triton.awq_fused_gemm import packed_int4_symmetric_fused_gemm_safe

                _awq_stat_inc("awq_fused_attempt")
                out, used_fused, _ = packed_int4_symmetric_fused_gemm_safe(
                    x.reshape(-1, x.shape[-1]).contiguous(),
                    self.qweight,
                    self.scales,
                    int(self.group_size),
                    bias=bias,
                )
                if used_fused:
                    _awq_stat_inc("awq_fused_success")
                    return out.view(*x.shape[:-1], out.shape[-1])
            except Exception:
                pass
            self._cached_fused_decision = False
            return _dense_fallback()

        if use_fused_cached is False:
            return _dense_fallback()

        if _should_use_high_fidelity_awq(self.prefix, self.high_fidelity):
            self._cached_fused_decision = False
            return _dense_fallback()

        if _env_awq_fused_gemm_force():
            self._cached_fused_decision = True
            return self.matmul(x, bias)

        policy = resolve_awq_execution_policy(self.prefix, x, self.profile_hint)
        use_fused = False
        if policy.prefer_fused or _env_awq_fused_gemm_force():
            try:
                from vllm.model_executor.layers.quantization.awq_triton import (
                    packed_int4_fused_capability_check,
                )

                use_fused, _ = packed_int4_fused_capability_check(
                    x, self.qweight, self.scales, int(self.group_size)
                )
            except Exception as e:
                print(
                    f">>>> DEBUG: PackedInt4 capability check EXCEPTION for {self.prefix}: {e}"
                )
        self._cached_fused_decision = bool(use_fused)
        if self._cached_fused_decision:
            return self.matmul(x, bias)
        return _dense_fallback()
