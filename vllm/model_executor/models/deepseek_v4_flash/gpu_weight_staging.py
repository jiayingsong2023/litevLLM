# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch

from .expert_cache import (
    DeepSeekV4FlashCacheAdmissionPolicy,
    DeepSeekV4FlashCacheKey,
    DeepSeekV4FlashExpertPrefetchRequest,
    DeepSeekV4FlashHotExpertPolicy,
)
from .gguf_reader import DeepSeekV4FlashTensor
from .weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashQuantizedExpertPayload,
)


class DeepSeekV4FlashGroupedExpertStore(Protocol):
    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor: ...

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload: ...

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor: ...

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview: ...


@dataclass(frozen=True)
class DeepSeekV4FlashStagedExpert:
    expert_id: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4FlashStagedQuantizedExpertPayload:
    tensor_name: str
    expert_id: int
    ggml_type: int
    rows: int
    columns: int
    payload: torch.Tensor


DeepSeekV4FlashSelectedExpertPayloads = tuple[
    int,
    DeepSeekV4FlashStagedQuantizedExpertPayload,
    DeepSeekV4FlashStagedQuantizedExpertPayload,
    DeepSeekV4FlashStagedQuantizedExpertPayload,
]


class DeepSeekV4FlashGPUWeightStager:
    """Stage decoded GGUF expert matrices into a per-device GPU cache."""

    def __init__(
        self,
        store: DeepSeekV4FlashGroupedExpertStore,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_staged_bytes: int | None = None,
        hot_expert_policy: DeepSeekV4FlashHotExpertPolicy | None = None,
        cache_admission_policy: DeepSeekV4FlashCacheAdmissionPolicy | None = None,
    ) -> None:
        self.store = store
        self.device = torch.device(device or "cuda")
        self.dtype = dtype
        self.hot_expert_policy = hot_expert_policy or DeepSeekV4FlashHotExpertPolicy()
        self.cache_admission_policy = (
            cache_admission_policy or DeepSeekV4FlashCacheAdmissionPolicy()
        )
        if max_staged_bytes is not None and max_staged_bytes < 0:
            raise ValueError("max staged bytes must be non-negative")
        self.max_staged_bytes = max_staged_bytes
        self._prefetch_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        self._full_resident_enabled = False
        self._staged_bytes = 0
        self._grouped_cache: dict[DeepSeekV4FlashCacheKey, torch.Tensor] = {}
        self._dynamic_cache: dict[DeepSeekV4FlashCacheKey, torch.Tensor] = {}
        self._cache_entry_bytes: dict[DeepSeekV4FlashCacheKey, int] = {}
        self._grouped_cache_experts: dict[
            DeepSeekV4FlashCacheKey,
            tuple[int | None, int],
        ] = {}
        self._lru_cache_keys: OrderedDict[DeepSeekV4FlashCacheKey, None] = OrderedDict()
        self._pinned_cache_keys: set[DeepSeekV4FlashCacheKey] = set()
        self._manual_pinned_experts: set[tuple[int, int]] = set()
        self._cache_stats = {
            "dynamic_hits": 0,
            "dynamic_misses": 0,
            "grouped_hits": 0,
            "grouped_misses": 0,
            "loaded_bytes": 0,
            "lru_evictions": 0,
            "prefetch_failures": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_payload_hits": 0,
            "prefetch_payload_misses": 0,
            "prefetch_payload_streamed_bytes": 0,
            "streamed_bytes": 0,
            "batched_payload_stage_calls": 0,
            "cpu_payload_cache_hits": 0,
            "cpu_payload_cache_misses": 0,
            "cpu_payload_cache_evictions": 0,
        }
        self._cpu_payload_cache: OrderedDict[tuple[str, int], torch.Tensor] = (
            OrderedDict()
        )
        self._cpu_payload_cache_bytes = 0
        self._cpu_payload_cache_capacity = self._read_cpu_payload_cache_capacity()

    @property
    def full_resident_enabled(self) -> bool:
        return self._full_resident_enabled

    def enable_full_resident_mode(self) -> None:
        self._full_resident_enabled = True
        self.max_staged_bytes = None

    @property
    def staged_bytes(self) -> int:
        return self._staged_bytes

    def memory_stats(self) -> dict[str, int | None]:
        return {
            "staged_bytes": self._staged_bytes,
            "max_staged_bytes": self.max_staged_bytes,
            "full_resident_enabled": int(self._full_resident_enabled),
            "dynamic_entries": len(self._dynamic_cache),
            "grouped_entries": len(self._grouped_cache),
            "pinned_entries": len(self._pinned_cache_keys),
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
    def _read_cpu_payload_cache_capacity() -> int:
        """Return the CPU payload cache capacity in bytes (default 0 = disabled)."""
        value = os.environ.get(
            "FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES", "0"
        )
        try:
            return max(0, int(value))
        except ValueError:
            warnings.warn(
                f"Ignoring malformed FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES "
                f"value {value!r}; using 0",
                stacklevel=2,
            )
            return 0

    def _get_cpu_cached_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor | None:
        if self._cpu_payload_cache_capacity <= 0:
            return None
        key = (tensor.name, expert_id)
        payload = self._cpu_payload_cache.get(key)
        if payload is None:
            return None
        self._cpu_payload_cache.move_to_end(key)
        self._cache_stats["cpu_payload_cache_hits"] += 1
        return payload

    def _put_cpu_cached_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
        payload: torch.Tensor,
    ) -> None:
        if self._cpu_payload_cache_capacity <= 0:
            return
        nbytes = payload.numel() * payload.element_size()
        if nbytes > self._cpu_payload_cache_capacity:
            return
        key = (tensor.name, expert_id)
        if key in self._cpu_payload_cache:
            self._cpu_payload_cache.move_to_end(key)
            return
        while (
            self._cpu_payload_cache
            and self._cpu_payload_cache_bytes + nbytes
            > self._cpu_payload_cache_capacity
        ):
            _, evicted = self._cpu_payload_cache.popitem(last=False)
            self._cpu_payload_cache_bytes -= (
                evicted.numel() * evicted.element_size()
            )
            self._cache_stats["cpu_payload_cache_evictions"] += 1
        self._cpu_payload_cache[key] = payload
        self._cpu_payload_cache_bytes += nbytes
        self._cache_stats["cpu_payload_cache_misses"] += 1

    def _record_prefetch_delta(self, before_stats: dict[str, int]) -> None:
        after_stats = self._cache_stats
        grouped_hits = after_stats["grouped_hits"] - before_stats["grouped_hits"]
        grouped_misses = after_stats["grouped_misses"] - before_stats["grouped_misses"]
        streamed_bytes = after_stats["streamed_bytes"] - before_stats["streamed_bytes"]
        self._cache_stats["prefetch_hits"] += grouped_hits
        self._cache_stats["prefetch_misses"] += grouped_misses
        self._cache_stats["prefetch_payload_hits"] += grouped_hits
        self._cache_stats["prefetch_payload_misses"] += grouped_misses
        self._cache_stats["prefetch_payload_streamed_bytes"] += streamed_bytes

    @staticmethod
    def _dtype_nbytes(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def _dynamic_cache_key(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
        extra: tuple[int | str, ...],
    ) -> DeepSeekV4FlashCacheKey:
        return DeepSeekV4FlashCacheKey(
            namespace="dynamic",
            name=tensor.name,
            device=str(self.device),
            dtype=str(dtype),
            extra=extra,
        )

    def _grouped_cache_key(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashCacheKey:
        return DeepSeekV4FlashCacheKey(
            namespace="grouped",
            name=tensor.name,
            device=str(self.device),
            dtype=str(self.dtype),
            extra=(expert_id,),
        )

    def _grouped_payload_cache_key(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashCacheKey:
        return DeepSeekV4FlashCacheKey(
            namespace="grouped",
            name=tensor.name,
            device=str(self.device),
            dtype=str(torch.uint8),
            extra=("raw_payload", expert_id),
        )

    def _record_lru_hit(self, cache_key: DeepSeekV4FlashCacheKey) -> None:
        if cache_key in self._lru_cache_keys:
            self._lru_cache_keys.move_to_end(cache_key)

    def _drop_cached_entry(
        self,
        cache_key: DeepSeekV4FlashCacheKey,
        *,
        count_eviction: bool,
    ) -> None:
        if cache_key.namespace == "grouped":
            self._grouped_cache.pop(cache_key, None)
            self._grouped_cache_experts.pop(cache_key, None)
        else:
            self._dynamic_cache.pop(cache_key, None)
        self._lru_cache_keys.pop(cache_key, None)
        self._pinned_cache_keys.discard(cache_key)
        self._staged_bytes -= self._cache_entry_bytes.pop(cache_key, 0)
        if count_eviction:
            self._cache_stats["lru_evictions"] += 1

    def _prepare_cache_insert(self, nbytes: int) -> bool:
        if nbytes < 0:
            raise ValueError("staged byte reservation must be non-negative")
        if self.max_staged_bytes is None:
            return True
        while self._staged_bytes + nbytes > self.max_staged_bytes:
            evictable_key = next(
                (
                    cache_key
                    for cache_key in self._lru_cache_keys
                    if cache_key not in self._pinned_cache_keys
                ),
                None,
            )
            if evictable_key is None:
                return False
            self._drop_cached_entry(evictable_key, count_eviction=True)
        return True

    def _register_cached_entry(
        self,
        cache_key: DeepSeekV4FlashCacheKey,
        tensor: torch.Tensor,
        nbytes: int,
        *,
        pinned: bool = False,
    ) -> None:
        if cache_key.namespace == "grouped":
            self._grouped_cache[cache_key] = tensor
        else:
            self._dynamic_cache[cache_key] = tensor
        self._cache_entry_bytes[cache_key] = nbytes
        self._lru_cache_keys[cache_key] = None
        self._lru_cache_keys.move_to_end(cache_key)
        if pinned:
            self._pinned_cache_keys.add(cache_key)
        self._staged_bytes += nbytes

    def _record_streamed_bytes(self, nbytes: int) -> None:
        self._cache_stats["streamed_bytes"] += int(nbytes)

    def _profile_section(self, name: str, **metadata: object) -> Iterator[None]:
        profiler = getattr(self, "profiler", None)
        section = getattr(profiler, "section", None)
        if not callable(section):
            return nullcontext()
        return section(name, **metadata)

    def _is_pinned_grouped_expert(
        self,
        layer_idx: int | None,
        expert_id: int,
    ) -> bool:
        if self._full_resident_enabled:
            return True
        if layer_idx is None:
            return False
        return (
            self.hot_expert_policy.is_pinned_expert(layer_idx, expert_id)
            or (layer_idx, expert_id) in self._manual_pinned_experts
        )

    def _should_cache_grouped_expert(
        self,
        layer_idx: int | None,
        expert_id: int,
    ) -> bool:
        if self._full_resident_enabled:
            return True
        return self.cache_admission_policy.should_cache_grouped_expert(
            layer_idx=layer_idx,
            expert_id=expert_id,
        )

    def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash matrix staging requires a CUDA device")
        cache_key = self._dynamic_cache_key(tensor, dtype=self.dtype, extra=("matrix",))
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            with self._profile_section(
                "stage_matrix",
                tensor=tensor.name,
                cache="hit",
            ):
                self._record_lru_hit(cache_key)
                self.record_cache_hit("dynamic", tensor_name=tensor.name)
                return cached

        with self._profile_section("stage_matrix", tensor=tensor.name, cache="miss"):
            decoded = self.store.decode_matrix(tensor)
            if decoded.ndim != 2:
                raise ValueError(f"matrix tensor must be 2-D; got {decoded.ndim}-D")
            nbytes = decoded.numel() * self._dtype_nbytes(self.dtype)
            if not self._prepare_cache_insert(nbytes):
                self._record_streamed_bytes(nbytes)
                return decoded.to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
            self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
            staged = decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
            self._register_cached_entry(cache_key, staged, nbytes)
            return staged

    def stage_vector(
        self,
        tensor: DeepSeekV4FlashTensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash vector staging requires a CUDA device")
        cache_key = self._dynamic_cache_key(tensor, dtype=dtype, extra=("vector",))
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            with self._profile_section(
                "stage_vector",
                tensor=tensor.name,
                cache="hit",
            ):
                self._record_lru_hit(cache_key)
                self.record_cache_hit("dynamic", tensor_name=tensor.name)
                return cached

        with self._profile_section("stage_vector", tensor=tensor.name, cache="miss"):
            decoded = self.store.tensor_to_torch(tensor, dtype=dtype)
            if decoded.ndim != 1:
                raise ValueError(f"vector tensor must be 1-D; got {decoded.ndim}-D")
            nbytes = decoded.numel() * self._dtype_nbytes(dtype)
            if not self._prepare_cache_insert(nbytes):
                self._record_streamed_bytes(nbytes)
                return decoded.to(device=self.device, dtype=dtype, non_blocking=True)
            self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
            staged = decoded.to(device=self.device, dtype=dtype, non_blocking=True)
            self._register_cached_entry(cache_key, staged, nbytes)
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
    ) -> tuple[DeepSeekV4FlashCacheKey, DeepSeekV4FlashCacheKey]:
        values_key = self._dynamic_cache_key(
            tensor,
            dtype=values_dtype,
            extra=("output_q8", row_start, row_end, "values"),
        )
        scales_key = self._dynamic_cache_key(
            tensor,
            dtype=torch.float32,
            extra=("output_q8", row_start, row_end, "scales"),
        )
        return values_key, scales_key

    def _output_q8_raw_cache_key(
        self,
        tensor: DeepSeekV4FlashTensor,
    ) -> DeepSeekV4FlashCacheKey:
        return self._dynamic_cache_key(
            tensor,
            dtype=torch.uint8,
            extra=("output_q8_raw", "payload"),
        )

    def stage_q8_raw_payload(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash raw Q8 staging requires a CUDA device")
        cache_key = self._output_q8_raw_cache_key(tensor)
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            with self._profile_section(
                "stage_q8_raw",
                tensor=tensor.name,
                cache="hit",
            ):
                self._record_lru_hit(cache_key)
                self.record_cache_hit("dynamic", tensor_name=tensor.name)
                return cached

        with self._profile_section("stage_q8_raw", tensor=tensor.name, cache="miss"):
            payload = self.store.tensor_payload(tensor)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The given buffer is not writable.*",
                        category=UserWarning,
                    )
                    raw = torch.frombuffer(payload, dtype=torch.uint8).clone()
            finally:
                payload.release()
            nbytes = raw.numel() * raw.element_size()
            if not self._prepare_cache_insert(nbytes):
                self._record_streamed_bytes(nbytes)
                return raw.to(device=self.device, non_blocking=True)
            staged = raw.to(device=self.device, non_blocking=True)
            self._register_cached_entry(cache_key, staged, nbytes)
            self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
            return staged

    def warm_static_decode_weights(
        self,
        *,
        output_weight: DeepSeekV4FlashTensor,
    ) -> None:
        self.stage_q8_raw_payload(output_weight)

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
            self._record_lru_hit(values_key)
            self._record_lru_hit(scales_key)
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
            self._record_lru_hit(values_key)
            self._record_lru_hit(scales_key)
            self.record_cache_hit("dynamic", tensor_name=tensor.name)
            return cached_values, cached_scales

        if cached_values is not None:
            self._drop_cached_entry(values_key, count_eviction=False)
        if cached_scales is not None:
            self._drop_cached_entry(scales_key, count_eviction=False)

        nbytes = values.numel() * values.element_size()
        nbytes += scales.numel() * self._dtype_nbytes(torch.float32)
        if not self._prepare_cache_insert(nbytes):
            self._record_streamed_bytes(nbytes)
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
        values_nbytes = values.numel() * values.element_size()
        scales_nbytes = scales.numel() * self._dtype_nbytes(torch.float32)
        self._register_cached_entry(values_key, staged_values, values_nbytes)
        self._register_cached_entry(scales_key, staged_scales, scales_nbytes)
        self.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
        return staged_values, staged_scales

    def clear_dynamic_cache(self) -> None:
        for cache_key in list(self._dynamic_cache):
            self._drop_cached_entry(cache_key, count_eviction=False)

    def pin_grouped_expert(self, layer_idx: int, expert_id: int) -> None:
        self._manual_pinned_experts.add((layer_idx, expert_id))
        for cache_key in list(self._grouped_cache):
            if self._grouped_cache_experts.get(cache_key) == (layer_idx, expert_id):
                self._pinned_cache_keys.add(cache_key)

    def stage_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash expert staging requires a CUDA device")
        should_cache = self._should_cache_grouped_expert(layer_idx, expert_id)
        cache_key = self._grouped_cache_key(tensor, expert_id)
        cached = self._grouped_cache.get(cache_key) if should_cache else None
        if cached is not None:
            if layer_idx is not None:
                self._grouped_cache_experts[cache_key] = (layer_idx, expert_id)
            if self._is_pinned_grouped_expert(layer_idx, expert_id):
                self._pinned_cache_keys.add(cache_key)
            self._record_lru_hit(cache_key)
            self.record_cache_hit("grouped", tensor_name=tensor.name)
            return cached

        decoded = self.store.decode_grouped_expert_matrix(tensor, expert_id)
        if decoded.ndim != 2:
            raise ValueError(f"grouped expert matrix must be 2-D; got {decoded.ndim}-D")
        nbytes = decoded.numel() * self._dtype_nbytes(self.dtype)
        if not should_cache or not self._prepare_cache_insert(nbytes):
            self._record_streamed_bytes(nbytes)
            return decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.record_cache_miss("grouped", nbytes, tensor_name=tensor.name)
        staged = decoded.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pinned = self._is_pinned_grouped_expert(layer_idx, expert_id)
        self._register_cached_entry(cache_key, staged, nbytes, pinned=pinned)
        self._grouped_cache_experts[cache_key] = (layer_idx, expert_id)
        return staged

    def stage_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ) -> DeepSeekV4FlashStagedQuantizedExpertPayload:
        """Return a staged raw payload tensor for a grouped expert.

        The returned ``payload`` tensor is a stable cached object for the
        ``(tensor, expert_id)`` pair: repeated calls return the same tensor
        instance (backed by ``_grouped_cache``).  Callers may therefore copy
        new expert bytes into the returned payload in place before graph
        replay without changing the tensor identity seen by a captured graph.
        """
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash expert staging requires a CUDA device")
        should_cache = self._should_cache_grouped_expert(layer_idx, expert_id)
        cache_key = self._grouped_payload_cache_key(tensor, expert_id)
        cached = self._grouped_cache.get(cache_key) if should_cache else None
        if cached is not None:
            if layer_idx is not None:
                self._grouped_cache_experts[cache_key] = (layer_idx, expert_id)
            if self._is_pinned_grouped_expert(layer_idx, expert_id):
                self._pinned_cache_keys.add(cache_key)
            self._record_lru_hit(cache_key)
            self.record_cache_hit("grouped", tensor_name=tensor.name)
            input_size, output_size, _expert_count = tensor.dims
            return DeepSeekV4FlashStagedQuantizedExpertPayload(
                tensor_name=tensor.name,
                expert_id=expert_id,
                ggml_type=tensor.tensor_type,
                rows=output_size,
                columns=input_size,
                payload=cached,
            )

        cpu_payload = self._get_cpu_cached_payload(tensor, expert_id)
        if cpu_payload is None:
            with self._profile_section(
                "raw_payload_read_clone",
                tensor=tensor.name,
                expert_id=expert_id,
                layer_idx=layer_idx,
            ):
                raw = self.store.raw_grouped_expert_payload(tensor, expert_id)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The given buffer is not writable.*",
                            category=UserWarning,
                        )
                        cpu_payload = torch.frombuffer(
                            raw.payload, dtype=torch.uint8
                        ).clone()
                finally:
                    raw.payload.release()
            self._put_cpu_cached_payload(tensor, expert_id, cpu_payload)
        nbytes = cpu_payload.numel() * cpu_payload.element_size()
        if not should_cache or not self._prepare_cache_insert(nbytes):
            self._record_streamed_bytes(nbytes)
            with self._profile_section(
                "h2d_copy_enqueue",
                tensor=tensor.name,
                expert_id=expert_id,
                layer_idx=layer_idx,
                cached=False,
            ):
                staged = cpu_payload.to(device=self.device, non_blocking=True)
            return DeepSeekV4FlashStagedQuantizedExpertPayload(
                tensor_name=raw.tensor_name,
                expert_id=raw.expert_id,
                ggml_type=raw.ggml_type,
                rows=raw.rows,
                columns=raw.columns,
                payload=staged,
            )
        self.record_cache_miss("grouped", nbytes, tensor_name=tensor.name)
        with self._profile_section(
            "h2d_copy_enqueue",
            tensor=tensor.name,
            expert_id=expert_id,
            layer_idx=layer_idx,
            cached=True,
        ):
            staged = cpu_payload.to(device=self.device, non_blocking=True)
        with self._profile_section(
            "cache_insert",
            tensor=tensor.name,
            expert_id=expert_id,
            layer_idx=layer_idx,
        ):
            pinned = self._is_pinned_grouped_expert(layer_idx, expert_id)
            self._register_cached_entry(cache_key, staged, nbytes, pinned=pinned)
            self._grouped_cache_experts[cache_key] = (layer_idx, expert_id)
        return DeepSeekV4FlashStagedQuantizedExpertPayload(
            tensor_name=raw.tensor_name,
            expert_id=raw.expert_id,
            ggml_type=raw.ggml_type,
            rows=raw.rows,
            columns=raw.columns,
            payload=staged,
        )

    def stage_grouped_expert_payloads_for_ids(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_ids: torch.Tensor,
        *,
        layer_idx: int | None = None,
    ) -> list[DeepSeekV4FlashSelectedExpertPayloads]:
        flattened = expert_ids.detach().reshape(-1)
        expert_id_values = [
            int(expert_id)
            for expert_id in flattened.to(device="cpu", dtype=torch.int64).tolist()
        ]
        self._cache_stats["batched_payload_stage_calls"] += 1
        staged_payloads: list[DeepSeekV4FlashSelectedExpertPayloads] = []
        for expert_id in expert_id_values:
            gate_payload = self.stage_grouped_expert_payload(
                tensors.gate,
                expert_id,
                layer_idx=layer_idx,
            )
            up_payload = self.stage_grouped_expert_payload(
                tensors.up,
                expert_id,
                layer_idx=layer_idx,
            )
            down_payload = self.stage_grouped_expert_payload(
                tensors.down,
                expert_id,
                layer_idx=layer_idx,
            )
            staged_payloads.append(
                (
                    expert_id,
                    gate_payload,
                    up_payload,
                    down_payload,
                )
            )
        return staged_payloads

    def copy_selected_expert_payload_bytes(
        self,
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
        expert_ids: torch.Tensor,
        *,
        layer_idx: int | None = None,
    ) -> list[DeepSeekV4FlashSelectedExpertPayloads]:
        """Stage payloads, ensuring returned tensors are the cached stable objects.

        Before graph replay, call this to copy the currently-selected expert bytes
        into the pre-allocated cached payload tensors.
        """
        return self.stage_grouped_expert_payloads_for_ids(
            grouped_experts,
            expert_ids,
            layer_idx=layer_idx,
        )

    def prefetch_grouped_experts(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if stream is None:
            self._prefetch_grouped_experts(tensors, request)
            return
        with torch.cuda.stream(stream):
            self._prefetch_grouped_experts(tensors, request)

    def prefetch_grouped_experts_async(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
    ) -> torch.cuda.Event:
        """Prefetch on a background stream and return an event for the compute stream.

        The caller must call wait_for_prefetch(event) on the compute stream before
        consuming the staged tensors. If the background stream is unavailable, the
        prefetch runs synchronously on the current stream and an already-completed
        event is still returned so callers can use a uniform wait path.
        """
        stream = self._prefetch_stream
        if stream is None:
            self._prefetch_grouped_experts(tensors, request)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            return event
        with torch.cuda.stream(stream):
            self._prefetch_grouped_experts(tensors, request)
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait_for_prefetch(
        self,
        event: torch.cuda.Event | None,
    ) -> None:
        """Make the current compute stream wait for a prefetch event."""
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

    def _prefetch_grouped_experts(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
    ) -> None:
        try:
            for expert_id in request.expert_ids:
                before_stats = self.cache_stats()
                try:
                    self.stage_grouped_expert_payload(
                        tensors.gate,
                        expert_id,
                        layer_idx=request.layer_idx,
                    )
                    self.stage_grouped_expert_payload(
                        tensors.up,
                        expert_id,
                        layer_idx=request.layer_idx,
                    )
                    self.stage_grouped_expert_payload(
                        tensors.down,
                        expert_id,
                        layer_idx=request.layer_idx,
                    )
                except (AttributeError, NotImplementedError):
                    self.stage_grouped_expert(
                        tensors,
                        expert_id,
                        layer_idx=request.layer_idx,
                    )
                finally:
                    self._record_prefetch_delta(before_stats)
        except Exception:
            self._cache_stats["prefetch_failures"] += 1
            raise

    def stage_grouped_expert(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ) -> DeepSeekV4FlashStagedExpert:
        return DeepSeekV4FlashStagedExpert(
            expert_id=expert_id,
            gate=self.stage_grouped_expert_matrix(
                tensors.gate,
                expert_id,
                layer_idx=layer_idx,
            ),
            up=self.stage_grouped_expert_matrix(
                tensors.up,
                expert_id,
                layer_idx=layer_idx,
            ),
            down=self.stage_grouped_expert_matrix(
                tensors.down,
                expert_id,
                layer_idx=layer_idx,
            ),
        )
