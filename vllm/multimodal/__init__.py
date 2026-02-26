# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .hasher import MultiModalHasher
from .inputs import (
    BatchedTensorInputs,
    ModalityData,
    MultiModalDataBuiltins,
    MultiModalDataDict,
    MultiModalKwargsItems,
    MultiModalPlaceholderDict,
    MultiModalUUIDDict,
    NestedTensors,
)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "MultiModalPlaceholderDict",
    "MultiModalUUIDDict",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
