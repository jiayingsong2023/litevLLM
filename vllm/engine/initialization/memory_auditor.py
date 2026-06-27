# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class MemoryAuditor:
    """Snapshot CUDA-resident model tensor footprint by dtype."""

    def __init__(self, *, device: torch.device, topn: int = 20) -> None:
        self.device = device
        self.topn = int(topn)

    def audit(self, model: torch.nn.Module) -> dict[str, Any]:
        device = self.device
        topn = self.topn
        param_total = 0
        buffer_total = 0
        param_count = 0
        buffer_count = 0
        param_dtype_bytes: dict[str, int] = {}
        buffer_dtype_bytes: dict[str, int] = {}
        param_rows: list[dict[str, Any]] = []
        buffer_rows: list[dict[str, Any]] = []

        for name, p in model.named_parameters():
            if not isinstance(p, torch.Tensor) or p.device != device:
                continue
            size = int(p.numel() * p.element_size())
            param_total += size
            param_count += 1
            key = _dtype_name(p.dtype)
            param_dtype_bytes[key] = param_dtype_bytes.get(key, 0) + size
            param_rows.append(
                {
                    "name": str(name),
                    "shape": tuple(int(x) for x in p.shape),
                    "dtype": key,
                    "bytes": int(size),
                }
            )

        for name, b in model.named_buffers():
            if not isinstance(b, torch.Tensor) or b.device != device:
                continue
            size = int(b.numel() * b.element_size())
            buffer_total += size
            buffer_count += 1
            key = _dtype_name(b.dtype)
            buffer_dtype_bytes[key] = buffer_dtype_bytes.get(key, 0) + size
            buffer_rows.append(
                {
                    "name": str(name),
                    "shape": tuple(int(x) for x in b.shape),
                    "dtype": key,
                    "bytes": int(size),
                }
            )

        awq_cache_bytes = 0
        try:
            from vllm.model_executor.layers.quantization.tensor import (
                get_awq_runtime_stats,
            )

            awq_stats = get_awq_runtime_stats()
            awq_cache_bytes = int(awq_stats.get("cache_bytes", 0) or 0)
        except Exception:
            awq_cache_bytes = 0

        return {
            "params_total_bytes": int(param_total),
            "buffers_total_bytes": int(buffer_total),
            "params_count": int(param_count),
            "buffers_count": int(buffer_count),
            "params_dtype_bytes": param_dtype_bytes,
            "buffers_dtype_bytes": buffer_dtype_bytes,
            "awq_cache_bytes": int(awq_cache_bytes),
            "topn": int(topn),
            "params_top": sorted(param_rows, key=lambda x: -int(x["bytes"]))[:topn],
            "buffers_top": sorted(buffer_rows, key=lambda x: -int(x["bytes"]))[:topn],
        }
