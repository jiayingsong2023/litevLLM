# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from .config import DEEPSEEK_V4_FLASH_SHAPE
from .gguf_reader import (
    DeepSeekV4FlashGGUF,
    DeepSeekV4FlashTensor,
    read_deepseek_v4_flash_gguf_from_view,
)


class DeepSeekV4FlashWeightStoreError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekV4FlashSemanticBindings:
    token_embedding: DeepSeekV4FlashTensor
    representative_layer_tensor: DeepSeekV4FlashTensor
    attention_query_by_layer: dict[int, DeepSeekV4FlashTensor]


@dataclass(frozen=True)
class DeepSeekV4FlashExpertCachePolicy:
    max_dynamic_bytes: int
    pinned_experts: tuple[tuple[int, int], ...] = ()
    defer_eviction_during_forward: bool = True

    def __post_init__(self) -> None:
        if self.max_dynamic_bytes < 0:
            raise ValueError("expert cache bytes must be non-negative")
        seen: set[tuple[int, int]] = set()
        for layer_idx, expert_id in self.pinned_experts:
            if layer_idx < 0 or layer_idx >= DEEPSEEK_V4_FLASH_SHAPE.num_layers:
                raise ValueError(f"layer index out of range: {layer_idx}")
            if expert_id < 0 or expert_id >= DEEPSEEK_V4_FLASH_SHAPE.num_experts:
                raise ValueError(f"expert id out of range: {expert_id}")
            key = (layer_idx, expert_id)
            if key in seen:
                raise ValueError(f"duplicate pinned experts are not allowed: {key}")
            seen.add(key)


@dataclass(frozen=True)
class DeepSeekV4FlashWeightStoreDiagnostics:
    tensor_count: int
    file_size_bytes: int
    mmap_size_bytes: int
    bound_tensor_count: int
    missing_required_semantic_tensors: tuple[str, ...]
    tensor_type_counts: dict[int, int]
    tensor_type_samples: dict[int, tuple[DeepSeekV4FlashTensor, ...]]
    unaligned_tensor_offsets: tuple[str, ...]


_REQUIRED_TENSORS: tuple[tuple[str, str], ...] = (
    ("token_embedding", "token_embd.weight"),
    ("layer_0_attention_query", "blk.0.attn_q.weight"),
)


class DeepSeekV4FlashWeightStore:
    def __init__(
        self,
        *,
        model: DeepSeekV4FlashGGUF,
        file: BinaryIO,
        mmap_obj: mmap.mmap,
        bindings: DeepSeekV4FlashSemanticBindings,
        diagnostics: DeepSeekV4FlashWeightStoreDiagnostics,
    ) -> None:
        self.model = model
        self._file = file
        self._mmap = mmap_obj
        self.bindings = bindings
        self.diagnostics = diagnostics

    @property
    def mmap_size_bytes(self) -> int:
        return self._mmap.size()

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> DeepSeekV4FlashWeightStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _bind_required_tensors(
    model: DeepSeekV4FlashGGUF,
) -> tuple[DeepSeekV4FlashSemanticBindings | None, tuple[str, ...]]:
    missing = tuple(
        semantic_name
        for semantic_name, tensor_name in _REQUIRED_TENSORS
        if tensor_name not in model.tensors
    )
    if missing:
        return None, missing

    attention_query_by_layer = _bind_attention_query_tensors(model)
    return (
        DeepSeekV4FlashSemanticBindings(
            token_embedding=model.tensors["token_embd.weight"],
            representative_layer_tensor=model.tensors["blk.0.attn_q.weight"],
            attention_query_by_layer=attention_query_by_layer,
        ),
        (),
    )


def _bind_attention_query_tensors(
    model: DeepSeekV4FlashGGUF,
) -> dict[int, DeepSeekV4FlashTensor]:
    bound: dict[int, DeepSeekV4FlashTensor] = {}
    prefix = "blk."
    suffix = ".attn_q.weight"
    for tensor in model.tensors.values():
        if not tensor.name.startswith(prefix) or not tensor.name.endswith(suffix):
            continue
        layer_text = tensor.name[len(prefix) : -len(suffix)]
        if not layer_text.isdigit():
            continue
        layer_idx = int(layer_text)
        if 0 <= layer_idx < DEEPSEEK_V4_FLASH_SHAPE.num_layers:
            bound[layer_idx] = tensor
    return bound


def _bound_tensor_count(
    bindings: DeepSeekV4FlashSemanticBindings | None,
) -> int:
    if bindings is None:
        return 0
    return 1 + len(bindings.attention_query_by_layer)


def _tensor_type_counts(model: DeepSeekV4FlashGGUF) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tensor in model.tensors.values():
        counts[tensor.tensor_type] = counts.get(tensor.tensor_type, 0) + 1
    return counts


def _tensor_type_samples(
    model: DeepSeekV4FlashGGUF,
    *,
    samples_per_type: int = 3,
) -> dict[int, tuple[DeepSeekV4FlashTensor, ...]]:
    samples: dict[int, list[DeepSeekV4FlashTensor]] = {}
    for tensor in model.tensors.values():
        type_samples = samples.setdefault(tensor.tensor_type, [])
        if len(type_samples) < samples_per_type:
            type_samples.append(tensor)
    return {
        tensor_type: tuple(type_samples)
        for tensor_type, type_samples in samples.items()
    }


def _unaligned_tensor_offsets(
    model: DeepSeekV4FlashGGUF,
    *,
    alignment: int = 32,
) -> tuple[str, ...]:
    return tuple(
        tensor.name
        for tensor in model.tensors.values()
        if tensor.offset % alignment != 0
    )


def open_deepseek_v4_flash_weight_store(path: Path | str) -> DeepSeekV4FlashWeightStore:
    gguf_path = Path(path)
    file = gguf_path.open("rb")
    try:
        mmap_obj = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        file.close()
        raise
    view = memoryview(mmap_obj)
    try:
        try:
            model = read_deepseek_v4_flash_gguf_from_view(gguf_path, view)
        finally:
            view.release()
        bindings, missing = _bind_required_tensors(model)
        diagnostics = DeepSeekV4FlashWeightStoreDiagnostics(
            tensor_count=len(model.tensors),
            file_size_bytes=gguf_path.stat().st_size,
            mmap_size_bytes=mmap_obj.size(),
            bound_tensor_count=_bound_tensor_count(bindings),
            missing_required_semantic_tensors=missing,
            tensor_type_counts=_tensor_type_counts(model),
            tensor_type_samples=_tensor_type_samples(model),
            unaligned_tensor_offsets=_unaligned_tensor_offsets(model),
        )
        if bindings is None:
            missing_list = ", ".join(
                f"{semantic_name} ({tensor_name})"
                for semantic_name, tensor_name in _REQUIRED_TENSORS
                if semantic_name in missing
            )
            raise DeepSeekV4FlashWeightStoreError(
                "missing required inspect-only tensors: " + missing_list
            )
        return DeepSeekV4FlashWeightStore(
            model=model,
            file=file,
            mmap_obj=mmap_obj,
            bindings=bindings,
            diagnostics=diagnostics,
        )
    except Exception:
        mmap_obj.close()
        file.close()
        raise
