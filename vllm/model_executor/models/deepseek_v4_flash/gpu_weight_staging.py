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
    ) -> None:
        self.store = store
        self.device = torch.device(device or "cuda")
        self.dtype = dtype
        self._grouped_cache: dict[tuple[str, int, str, torch.dtype], torch.Tensor] = {}
        self._dynamic_cache: dict[
            tuple[str, str, torch.dtype, str],
            torch.Tensor,
        ] = {}

    def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("DeepSeek V4 Flash matrix staging requires a CUDA device")
        cache_key = (tensor.name, str(self.device), self.dtype, "matrix")
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            return cached

        decoded = self.store.decode_matrix(tensor)
        if decoded.ndim != 2:
            raise ValueError(f"matrix tensor must be 2-D; got {decoded.ndim}-D")
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
            return cached

        decoded = self.store.tensor_to_torch(tensor, dtype=dtype)
        if decoded.ndim != 1:
            raise ValueError(f"vector tensor must be 1-D; got {decoded.ndim}-D")
        staged = decoded.to(device=self.device, dtype=dtype, non_blocking=True)
        self._dynamic_cache[cache_key] = staged
        return staged

    def clear_dynamic_cache(self) -> None:
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
            return cached

        decoded = self.store.decode_grouped_expert_matrix(tensor, expert_id)
        if decoded.ndim != 2:
            raise ValueError(
                f"grouped expert matrix must be 2-D; got {decoded.ndim}-D"
            )
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
