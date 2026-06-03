# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

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


@dataclass(frozen=True)
class DeepSeekV4FlashWeightStoreDiagnostics:
    tensor_count: int
    file_size_bytes: int
    mmap_size_bytes: int
    bound_tensor_count: int
    missing_required_semantic_tensors: tuple[str, ...]


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

    return (
        DeepSeekV4FlashSemanticBindings(
            token_embedding=model.tensors["token_embd.weight"],
            representative_layer_tensor=model.tensors["blk.0.attn_q.weight"],
        ),
        (),
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
            bound_tensor_count=sum(
                1
                for _, tensor_name in _REQUIRED_TENSORS
                if tensor_name in model.tensors
            ),
            missing_required_semantic_tensors=missing,
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
