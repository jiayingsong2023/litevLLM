# SPDX-License-Identifier: Apache-2.0
"""
Optional AWQ weight cache prefetch: materialize FP8/block or dense LRU entries once
so steady-state matmul uses BLAS (same as non-fused path) while weights stay packed on disk.

Uses each LiteLinear's quant_config.apply(..., dummy_input) with fused disabled so
PackedInt4Weight / AWQWeight matmul runs the dequant+cache path and populates LRU.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.lite_linear import LiteLinear


def _env_prefetch_awq_cache() -> bool:
    return os.environ.get("FASTINFERENCE_AWQ_PREFETCH_WEIGHT_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def prefetch_awq_weight_caches(model: nn.Module) -> int:
    """
    Walk ``model`` and invoke ``quant_config.apply`` once per loaded quantized
    ``LiteLinear`` to populate ``LRUWeightCache`` (same as fused-off inference).

    Returns number of layers touched. Skips layers that fail or are not ready.
    """
    if not torch.cuda.is_available():
        return 0
    old_fused = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM")
    old_force = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_FORCE")
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM"] = "0"
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_FORCE"] = "0"
    count = 0
    try:
        for module in model.modules():
            if not isinstance(module, LiteLinear):
                continue
            qc: Any = getattr(module, "quant_config", None)
            if qc is None or not hasattr(qc, "apply"):
                continue
            qweight = getattr(module, "qweight", None)
            if qweight is None or qweight.numel() <= 1:
                continue
            try:
                device = qweight.device
                in_features = int(getattr(module, "input_size", 0) or 0)
                if in_features <= 0:
                    in_features = int(qweight.shape[1]) * 8
                dummy = torch.zeros((1, in_features), device=device, dtype=torch.bfloat16)
                qc.apply(module, dummy)
                count += 1
            except Exception:
                continue
        return count
    finally:
        if old_fused is None:
            os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM", None)
        else:
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM"] = old_fused
        if old_force is None:
            os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM_FORCE", None)
        else:
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_FORCE"] = old_force


def maybe_prefetch_awq_weight_caches(model: nn.Module) -> None:
    if not _env_prefetch_awq_cache():
        return
    n = prefetch_awq_weight_caches(model)
    if n > 0:
        print(
            f">>>> LiteEngine: AWQ weight cache prefetch: materialized {n} quantized layer(s) "
            f"(FASTINFERENCE_AWQ_PREFETCH_WEIGHT_CACHE=1)"
        )
