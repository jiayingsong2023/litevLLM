# SPDX-License-Identifier: Apache-2.0
"""
Shared helper for fusing two structurally-identical ``LiteLinear`` projections
into a single quantized GEMM.

Canonical callers:
  * Qwen3.5 linear-attention in_proj_a + in_proj_b
  * Qwen3.5 full-attention k_proj + v_proj
  * Qwen3.5 / Gemma4 MLP gate_proj + up_proj

Fusion rules:
  - ``lin_a`` and ``lin_b`` must share input_size, output_size, group_size,
    and bias presence.
  - Symmetric packed int4 (``PackedInt4Weight``) with ``qzeros is None``
    is supported.
  - Classic AWQ (``AWQWeight``) with matching ``qzeros`` shapes is supported.
  - ``force_high_fidelity_awq`` on either layer disables fusion (keeps the
    strict per-layer scale/zero math).
  - When LoRA is *active* (``lora_mapping`` contains at least one non-None
    adapter id), fusion is disabled so LoRA adapters retain per-branch
    control; the caller falls back to two independent matmuls. A bare
    ``[None, ...]`` list (non-LoRA request with the batch-builder boilerplate)
    is treated as inactive.

The fused weight object is attached to ``owner`` as an attribute named
``_fused_awq_pair_<cache_key>`` so re-entering the same module reuses it
without re-concatenating int4 bytes per step.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.lora.mapping import LoRAMapping
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.tensor import (
    AWQWeight,
    PackedInt4Weight,
    record_awq_audit_event,
)


def _lora_is_active(lora_mapping: Any) -> bool:
    """Return True only when at least one request in the batch actually
    carries a LoRA adapter id. The input_batch_builder always fills
    ``attn_metadata["lora_mapping"]`` with a per-request list; for plain
    (non-LoRA) requests that list is ``[None, None, ...]`` which must NOT
    disable fusion."""
    if lora_mapping is None:
        return False
    if isinstance(lora_mapping, LoRAMapping):
        return lora_mapping.adapter_count > 0
    if isinstance(lora_mapping, (list, tuple)):
        return any(x is not None for x in lora_mapping)
    return bool(lora_mapping)


def try_fused_awq_pair_matmul(
    x: torch.Tensor,
    lin_a: LiteLinear,
    lin_b: LiteLinear,
    owner: nn.Module,
    cache_key: str,
    *,
    lora_mapping: Any = None,
    inf_config: Any = None,
) -> torch.Tensor | None:
    """Attempt a single quantized matmul that yields the concatenation of
    ``lin_a(x)`` and ``lin_b(x)`` along the last dimension.

    Returns a ``(..., 2 * output_size)`` tensor on success, or ``None`` when
    the layers are structurally incompatible / fusion is disabled. Callers
    are expected to fall back to two separate forwards when ``None`` is
    returned.
    """
    # LoRA-aware branch: fusion concatenates quantized weight tensors, which
    # is incompatible with per-branch LoRA deltas. Skip and let caller run
    # the two projections independently.
    if _lora_is_active(lora_mapping):
        return None

    if lin_a.input_size != lin_b.input_size:
        return None
    if lin_a.output_size != lin_b.output_size:
        return None

    ba = getattr(lin_a, "bias", None)
    bb = getattr(lin_b, "bias", None)
    fused_bias: torch.Tensor | None = None
    if ba is not None and bb is not None:
        fused_bias = torch.cat([ba.reshape(-1), bb.reshape(-1)], dim=0).contiguous()
    elif ba is not None or bb is not None:
        # Mixed bias presence: punt to unfused path.
        return None

    if getattr(lin_a, "force_high_fidelity_awq", False) or getattr(
        lin_b, "force_high_fidelity_awq", False
    ):
        return None

    qwa = getattr(lin_a, "qweight", None)
    qwb = getattr(lin_b, "qweight", None)
    if qwa is None or qwb is None or qwa.numel() <= 1 or qwb.numel() <= 1:
        return None
    if tuple(qwa.shape) != tuple(qwb.shape):
        return None
    if tuple(lin_a.scales.shape) != tuple(lin_b.scales.shape):
        return None

    gs = int(getattr(lin_a, "group_size", 128))
    if int(getattr(lin_b, "group_size", 128)) != gs:
        return None

    za = getattr(lin_a, "qzeros", None)
    zb = getattr(lin_b, "qzeros", None)
    has_awq_zeros = (
        za is not None
        and zb is not None
        and za.numel() > 1
        and zb.numel() > 1
        and tuple(za.shape) == tuple(zb.shape)
    )
    has_symmetric_packed = (za is None or za.numel() <= 1) and (
        zb is None or zb.numel() <= 1
    )
    if not has_awq_zeros and not has_symmetric_packed:
        return None

    cache_attr = f"_fused_awq_pair_{cache_key}"
    fused_w = getattr(owner, cache_attr, None)
    if fused_w is None:
        q_cat = torch.cat([lin_a.qweight, lin_b.qweight], dim=0).contiguous()
        s_cat = torch.cat([lin_a.scales, lin_b.scales], dim=0).contiguous()
        if has_awq_zeros:
            fused_w = AWQWeight(
                q_cat,
                s_cat,
                torch.cat([lin_a.qzeros, lin_b.qzeros], dim=0).contiguous(),
                group_size=gs,
                prefix=getattr(lin_a, "prefix", "") or f"fused_pair:{cache_key}",
                high_fidelity=False,
                profile_hint=str(getattr(lin_a, "awq_profile_hint", "") or ""),
            )
        else:
            original_shape = getattr(lin_a, "weight_shape", None)
            if original_shape is not None:
                original_shape = (int(original_shape[0]) * 2, int(original_shape[1]))
            fused_w = PackedInt4Weight(
                q_cat,
                s_cat,
                group_size=gs,
                original_shape=original_shape,
                prefix=getattr(lin_a, "prefix", "") or f"fused_pair:{cache_key}",
                high_fidelity=False,
                profile_hint=str(getattr(lin_a, "awq_profile_hint", "") or ""),
            )
        setattr(owner, cache_attr, fused_w)

    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    out2 = fused_w.matmul(x2, fused_bias, config=inf_config)
    od = int(lin_a.output_size)
    return out2.reshape(*lead_shape, 2 * od)


def try_fused_awq_gate_up_activation(
    x: torch.Tensor,
    gate_proj: LiteLinear,
    up_proj: LiteLinear,
    *,
    activation: str,
    lora_mapping: Any = None,
    inf_config: Any = None,
) -> torch.Tensor | None:
    """Attempt decode-only fused gate/up GEMV that returns ``act(gate) * up``.

    This is intentionally narrower than ``try_fused_awq_pair_matmul``: it only
    handles symmetric packed-int4, M=1, no-bias Gemma-style MLP projections.
    Wider/prefill batches and classic AWQ fall back to existing pair fusion.
    """
    if _lora_is_active(lora_mapping):
        return None
    if gate_proj.input_size != up_proj.input_size:
        return None
    if gate_proj.output_size != up_proj.output_size:
        return None
    if (
        getattr(gate_proj, "bias", None) is not None
        or getattr(up_proj, "bias", None) is not None
    ):
        return None
    if getattr(gate_proj, "force_high_fidelity_awq", False) or getattr(
        up_proj, "force_high_fidelity_awq", False
    ):
        return None
    za = getattr(gate_proj, "qzeros", None)
    zb = getattr(up_proj, "qzeros", None)
    has_symmetric_packed = (za is None or za.numel() <= 1) and (
        zb is None or zb.numel() <= 1
    )
    if not has_symmetric_packed:
        return None
    qwg = getattr(gate_proj, "qweight", None)
    qwu = getattr(up_proj, "qweight", None)
    if qwg is None or qwu is None or qwg.numel() <= 1 or qwu.numel() <= 1:
        return None
    if tuple(qwg.shape) != tuple(qwu.shape):
        return None
    if tuple(gate_proj.scales.shape) != tuple(up_proj.scales.shape):
        return None
    gs = int(getattr(gate_proj, "group_size", 128))
    if int(getattr(up_proj, "group_size", 128)) != gs:
        return None
    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if int(x2.shape[0]) != 1:
        return None
    try:
        from vllm.kernels.triton.awq_fused_gemm import (
            packed_int4_symmetric_fused_gate_up_m1_safe,
        )

        out2, used, _ = packed_int4_symmetric_fused_gate_up_m1_safe(
            x2.contiguous(),
            qwg,
            qwu,
            gate_proj.scales,
            up_proj.scales,
            gs,
            activation=activation,
            config=inf_config,
        )
        if not used:
            return None
        return out2.reshape(*lead_shape, int(gate_proj.output_size))
    except Exception:
        return None


def try_fused_awq_mlp_streaming(
    x: torch.Tensor,
    gate_proj: LiteLinear,
    up_proj: LiteLinear,
    down_proj: LiteLinear,
    *,
    activation: str,
    lora_mapping: Any = None,
    inf_config: Any = None,
    prefix: str = "",
) -> torch.Tensor | None:
    """Attempt full Gemma-style AWQ MLP streaming fusion.

    This P2 helper is deliberately default-off. Until an isolated Triton
    candidate beats the current two-stage path, it only centralizes structural
    guards and audit visibility while preserving the existing fallback path.
    """
    audit_prefix = prefix or str(getattr(gate_proj, "prefix", "mlp"))

    def record_fallback(reason: str) -> None:
        shape = {
            "m": int(x.reshape(-1, x.shape[-1]).shape[0]) if x.dim() >= 2 else -1,
            "hidden": int(
                getattr(gate_proj, "input_size", x.shape[-1] if x.dim() else -1)
            ),
            "intermediate": int(getattr(gate_proj, "output_size", -1)),
            "out": int(getattr(down_proj, "output_size", -1)),
        }
        record_awq_audit_event(
            audit_prefix,
            "mlp_streaming_fallback",
            shape=shape,
            reason=reason,
        )

    try:
        from vllm.kernels.triton.awq_fused_gemm import (
            awq_mlp_streaming_fusion_enabled,
        )

        if not awq_mlp_streaming_fusion_enabled(config=inf_config):
            record_fallback("disabled")
            return None
    except Exception:
        record_fallback("policy_error")
        return None

    record_awq_audit_event(
        audit_prefix,
        "mlp_streaming_attempt",
        shape={
            "m": int(x.reshape(-1, x.shape[-1]).shape[0]) if x.dim() >= 2 else -1,
            "hidden": int(
                getattr(gate_proj, "input_size", x.shape[-1] if x.dim() else -1)
            ),
            "intermediate": int(getattr(gate_proj, "output_size", -1)),
            "out": int(getattr(down_proj, "output_size", -1)),
        },
        reason="policy_enabled",
    )
    if _lora_is_active(lora_mapping):
        record_fallback("lora_active")
        return None
    if x.dim() < 2 or int(x.reshape(-1, x.shape[-1]).shape[0]) != 1:
        record_fallback("input_not_m1")
        return None
    if gate_proj.input_size != up_proj.input_size:
        record_fallback("gate_up_input_mismatch")
        return None
    if gate_proj.output_size != up_proj.output_size:
        record_fallback("gate_up_output_mismatch")
        return None
    if down_proj.input_size != gate_proj.output_size:
        record_fallback("down_input_mismatch")
        return None
    if (
        getattr(gate_proj, "bias", None) is not None
        or getattr(up_proj, "bias", None) is not None
    ):
        record_fallback("gate_up_bias")
        return None
    if getattr(down_proj, "bias", None) is not None:
        record_fallback("down_bias")
        return None
    if int(getattr(gate_proj, "group_size", 128)) != 32:
        record_fallback("gate_group_size_not_32")
        return None
    if int(getattr(up_proj, "group_size", 128)) != 32:
        record_fallback("up_group_size_not_32")
        return None
    if int(getattr(down_proj, "group_size", 128)) != 32:
        record_fallback("down_group_size_not_32")
        return None
    za = getattr(gate_proj, "qzeros", None)
    zb = getattr(up_proj, "qzeros", None)
    zd = getattr(down_proj, "qzeros", None)
    if not all(z is None or z.numel() <= 1 for z in (za, zb, zd)):
        record_fallback("asymmetric_awq")
        return None
    del activation
    record_fallback("requires_cross_program_sharing")
    return None
