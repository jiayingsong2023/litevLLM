# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .gguf_reader import DeepSeekV4FlashTensor
from .weight_store import DeepSeekV4FlashGroupedExpertTensors


class DeepSeekV4FlashGroupedExpertStore(Protocol):
    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor: ...

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor: ...

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DeepSeekV4FlashStagedExpert:
    expert_id: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


class DeepSeekV4FlashGPUWeightStager:
    """Stage decoded GGUF expert matrices into a per-device GPU cache."""

    def __init__(
        self,
        store: DeepSeekV4FlashGroupedExpertStore,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_staged_bytes: int | None = None,
    ) -> None:
        self.store = store
        self.device = torch.device(device or "cuda")
        self.dtype = dtype
        if max_staged_bytes is not None and max_staged_bytes < 0:
            raise ValueError("max staged bytes must be non-negative")
        self.max_staged_bytes = max_staged_bytes
        self._staged_bytes = 0
        self._grouped_cache: dict[tuple[str, int, str, torch.dtype], torch.Tensor] = {}
        self._dynamic_cache: dict[
            tuple[str, str, torch.dtype, str],
            torch.Tensor,
        ] = {}
        self._cache_stats = {
            "dynamic_hits": 0,
            "dynamic_misses": 0,
            "grouped_hits": 0,
            "grouped_misses": 0,
            "loaded_bytes": 0,
        }

    @property
    def staged_bytes(self) -> int:
        return self._staged_bytes

    def memory_stats(self) -> dict[str, int | None]:
        return {
            "staged_bytes": self._staged_bytes,
            "max_staged_bytes": self.max_staged_bytes,
            "dynamic_entries": len(self._dynamic_cache),
            "grouped_entries": len(self._grouped_cache),
            **self.cache_stats(),
        }

    def record_cache_hit(self, cache_name: str, *, tensor_name: str) -> None:
        del tensor_name
        key = f"{cache_name}_hits"
        if key not in self._cache_stats:
            raise ValueError(f"unknown cache stats bucket: {cache_name}")
        self._cache_stats[key] += 1

    def record_cache_miss(
        self,
        cache_name: str,
        loaded_bytes: int,
        *,
        tensor_name: str,
    ) -> None:
        del tensor_name
        if loaded_bytes < 0:
            raise ValueError("loaded bytes must be non-negative")
        key = f"{cache_name}_misses"
        if key not in self._cache_stats:
            raise ValueError(f"unknown cache stats bucket: {cache_name}")
        self._cache_stats[key] += 1
        self._cache_stats["loaded_bytes"] += int(loaded_bytes)

    def cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    @staticmethod
    def _dtype_nbytes(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def _reserve_staged_bytes(self, nbytes: int, *, tensor_name: str) -> None:
        if nbytes < 0:
            raise ValueError("staged byte reservation must be non-negative")
        if self.max_staged_bytes is None:
            self._staged_bytes += nbytes
            return
        next_bytes = self._staged_bytes + nbytes
        if next_bytes > self.max_staged_bytes:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU staging cache exceeds memory budget: "
                f"tensor={tensor_name}, requested={nbytes} bytes, "
                f"resident={self._staged_bytes} bytes, "
                f"budget={self.max_staged_bytes} bytes"
            )
        self._staged_bytes = next_bytes

    def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash matrix staging requires a CUDA device")
        cache_key = (tensor.name, str(self.device), self.dtype, "matrix")
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            self.record_cache_hit("dynamic", tensor_name=tensor.name)
            return cached

        decoded = self.store.decode_matrix(tensor)
        if decoded.ndim != 2:
            raise ValueError(f"matrix tensor must be 2-D; got {decoded.ndim}-D")
        nbytes = decoded.numel() * self._dtype_nbytes(self.dtype)
        try:
            self._reserve_staged_bytes(nbytes, tensor_name=tensor.name)
        except RuntimeError:
            return decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
        staged = decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self._dynamic_cache[cache_key] = staged
        return staged

    def stage_vector(
        self,
        tensor: DeepSeekV4FlashTensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash vector staging requires a CUDA device")
        cache_key = (tensor.name, str(self.device), dtype, "vector")
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            self.record_cache_hit("dynamic", tensor_name=tensor.name)
            return cached

        decoded = self.store.tensor_to_torch(tensor, dtype=dtype)
        if decoded.ndim != 1:
            raise ValueError(f"vector tensor must be 1-D; got {decoded.ndim}-D")
        nbytes = decoded.numel() * self._dtype_nbytes(dtype)
        try:
            self._reserve_staged_bytes(nbytes, tensor_name=tensor.name)
        except RuntimeError:
            return decoded.to(device=self.device, dtype=dtype, non_blocking=True)
        self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
        staged = decoded.to(device=self.device, dtype=dtype, non_blocking=True)
        self._dynamic_cache[cache_key] = staged
        return staged

    @staticmethod
    def _validate_output_q8_row_range(
        tensor: DeepSeekV4FlashTensor,
        *,
        row_start: int,
        row_end: int,
    ) -> None:
        if row_start < 0 or row_end <= row_start:
            raise ValueError(
                "output Q8 chunk row range must satisfy "
                f"row_start >= 0 and row_end > row_start; "
                f"got row_start={row_start}, row_end={row_end}"
            )
        if len(tensor.dims) >= 2:
            rows = tensor.dims[1]
            if row_end > rows:
                raise ValueError(
                    "output Q8 chunk row range exceeds tensor rows; "
                    f"got row_start={row_start}, row_end={row_end}, rows={rows}"
                )

    def _output_q8_chunk_cache_keys(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        row_start: int,
        row_end: int,
        values_dtype: torch.dtype = torch.int8,
    ) -> tuple[
        tuple[str, str, torch.dtype, str],
        tuple[str, str, torch.dtype, str],
    ]:
        row_range = f"output_q8:{row_start}:{row_end}"
        values_key = (
            tensor.name,
            str(self.device),
            values_dtype,
            f"{row_range}:values",
        )
        scales_key = (
            tensor.name,
            str(self.device),
            torch.float32,
            f"{row_range}:scales",
        )
        return values_key, scales_key

    def get_output_q8_chunk(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        row_start: int,
        row_end: int,
        record_hit: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.device.type != "cuda":
            raise ValueError(
                "DeepSeek V4 Flash output Q8 staging requires a CUDA device"
            )
        self._validate_output_q8_row_range(
            tensor,
            row_start=row_start,
            row_end=row_end,
        )
        values_key, scales_key = self._output_q8_chunk_cache_keys(
            tensor,
            row_start=row_start,
            row_end=row_end,
        )
        cached_values = self._dynamic_cache.get(values_key)
        cached_scales = self._dynamic_cache.get(scales_key)
        if cached_values is None or cached_scales is None:
            return None
        if record_hit:
            self.record_cache_hit("dynamic", tensor_name=tensor.name)
        return cached_values, cached_scales

    def stage_output_q8_chunk(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        row_start: int,
        row_end: int,
        values: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type != "cuda":
            raise ValueError(
                "DeepSeek V4 Flash output Q8 staging requires a CUDA device"
            )
        self._validate_output_q8_row_range(
            tensor,
            row_start=row_start,
            row_end=row_end,
        )

        values_key, scales_key = self._output_q8_chunk_cache_keys(
            tensor,
            row_start=row_start,
            row_end=row_end,
            values_dtype=values.dtype,
        )
        cached_values = self._dynamic_cache.get(values_key)
        cached_scales = self._dynamic_cache.get(scales_key)
        if cached_values is not None and cached_scales is not None:
            self.record_cache_hit("dynamic", tensor_name=tensor.name)
            return cached_values, cached_scales

        partial_cached_bytes = 0
        if cached_values is not None:
            partial_cached_bytes += cached_values.nbytes
            del self._dynamic_cache[values_key]
        if cached_scales is not None:
            partial_cached_bytes += cached_scales.nbytes
            del self._dynamic_cache[scales_key]
        self._staged_bytes -= partial_cached_bytes

        nbytes = values.numel() * values.element_size()
        nbytes += scales.numel() * self._dtype_nbytes(torch.float32)
        try:
            self._reserve_staged_bytes(nbytes, tensor_name=tensor.name)
        except RuntimeError:
            return (
                values.to(device=self.device, non_blocking=True),
                scales.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                ),
            )
        staged_values = values.to(device=self.device, non_blocking=True)
        staged_scales = scales.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        self._dynamic_cache[values_key] = staged_values
        self._dynamic_cache[scales_key] = staged_scales
        self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
        return staged_values, staged_scales

    def clear_dynamic_cache(self) -> None:
        self._staged_bytes -= sum(
            tensor.nbytes for tensor in self._dynamic_cache.values()
        )
        self._dynamic_cache.clear()

    def stage_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash expert staging requires a CUDA device")
        cache_key = (tensor.name, expert_id, str(self.device), self.dtype)
        cached = self._grouped_cache.get(cache_key)
        if cached is not None:
            self.record_cache_hit("grouped", tensor_name=tensor.name)
            return cached

        decoded = self.store.decode_grouped_expert_matrix(tensor, expert_id)
        if decoded.ndim != 2:
            raise ValueError(
                f"grouped expert matrix must be 2-D; got {decoded.ndim}-D"
            )
        nbytes = decoded.numel() * self._dtype_nbytes(self.dtype)
        try:
            self._reserve_staged_bytes(nbytes, tensor_name=tensor.name)
        except RuntimeError:
            return decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.record_cache_miss("grouped", nbytes, tensor_name=tensor.name)
        staged = decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self._grouped_cache[cache_key] = staged
        return staged

    def stage_grouped_expert(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_id: int,
    ) -> DeepSeekV4FlashStagedExpert:
        return DeepSeekV4FlashStagedExpert(
            expert_id=expert_id,
            gate=self.stage_grouped_expert_matrix(tensors.gate, expert_id),
            up=self.stage_grouped_expert_matrix(tensors.up, expert_id),
            down=self.stage_grouped_expert_matrix(tensors.down, expert_id),
        )
