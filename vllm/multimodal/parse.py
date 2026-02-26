# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import numpy as np
import torch
from typing_extensions import assert_never

from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader

from .audio import AudioResampler, AudioSpec, normalize_audio
from .inputs import (
    AudioItem,
    HfAudioItem,
    HfImageItem,
    HfVideoItem,
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from .media import MediaWithBytes

_T = TypeVar("_T")
_I = TypeVar("_I")

if TYPE_CHECKING:
    import PIL.Image as PILImage
else:
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")

class ModalityDataItems(ABC, Generic[_T, _I]):

    def __init__(self, data: _T, modality: str) -> None:
        super().__init__()

        self.data: _T = data
        self.modality = modality

    def __repr__(self) -> str:
        return f"{type(self).__name__}(modality={self.modality!r}, len={len(self)})"

    def __len__(self) -> int:
        return self.get_count()

    def __getitem__(self, index: int) -> _I:
        return self.get(index)

    if TYPE_CHECKING:
        # Auto-generated
        def __iter__(self) -> Iterator[_I]: ...

    @abstractmethod
    def get_count(self) -> int:
        raise NotImplementedError

    def get_all(self) -> list[_I]:
        raise NotImplementedError

    @abstractmethod
    def get_passthrough_data(self) -> Mapping[str, object]:

    def _unwrap(self, item: _T | MediaWithBytes[_T]) -> _T:

    Single embeddings should be 2D (seq_len, hidden_size).
    Batched embeddings should be 3D (batch, seq_len, hidden_size).

    Args:
        tensor: The tensor to validate.
        modality: The modality name for error messages (e.g., "image", "audio").
        index: Optional index for list items, included in error messages.
    Base class for data items that are expressed as a batched embedding tensor,
    or a list of embedding tensors (one per item).
        if isinstance(self.data, torch.Tensor):
            validate_embedding_ndim(self.data, self.modality)
        else:
            # List of tensors: each should be 2D (seq_len, hidden_size)
            for idx, tensor in enumerate(self.data):
                if tensor.ndim != 2:
                    raise ValueError(
                        f"{self.modality.capitalize()} embedding [{idx}] must be "
                        f"2D (seq_len, hidden_size), got {tensor.ndim}D tensor "
                        f"with shape {tuple(tensor.shape)}"
                    )

    def _validate_hidden_size(self, expected_hidden_size: int) -> None:
        if isinstance(self.data, torch.Tensor):
            # Batched tensor: shape is (batch, seq_len, hidden_size)
            actual_hidden_size = self.data.shape[-1]
            if actual_hidden_size != expected_hidden_size:
                raise ValueError(
                    f"{self.modality.capitalize()} embedding hidden dimension "
                    f"mismatch: got {actual_hidden_size}, but model expects "
                    f"{expected_hidden_size}. Embedding shape: {tuple(self.data.shape)}"
                )
        else:
            # List of tensors: each has shape (seq_len, hidden_size)
            for idx, tensor in enumerate(self.data):
                actual_hidden_size = tensor.shape[-1]
                if actual_hidden_size != expected_hidden_size:
                    raise ValueError(
                        f"{self.modality.capitalize()} embedding [{idx}] hidden "
                        f"dimension mismatch: got {actual_hidden_size}, but model "
                        f"expects {expected_hidden_size}. "
                        f"Embedding shape: {tuple(tensor.shape)}"
                    )

    def _unwrap(
        self, item: torch.Tensor | MediaWithBytes[torch.Tensor]
    ) -> torch.Tensor:
    Base class for data items that are expressed as a dictionary of tensors.

    Usually, the dictionary keys correspond to the outputs of HF processor.

    def __init__(self, data: Sequence[Any]) -> None:
        super().__init__(data, "vision_chunk")

_D = TypeVar("_D", bound=ModalityDataItems[Any, Any])

class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):

    def get_count(self, modality: str, *, strict: bool = True) -> int:
        if modality not in self:
            if strict:
                available_modalities = set(self.keys())
                raise KeyError(
                    f"Modality {modality!r} not found. "
                    f"Available modalities: {available_modalities}"
                )

            return 0

        return self[modality].get_count()

    def get_all_counts(self) -> Mapping[str, int]:
        Get the data items belonging to a modality,
        requiring that they belong to a certain type.
    Parses [`MultiModalDataDict`][vllm.multimodal.inputs.MultiModalDataDict]
    into [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems].

    Args:
        target_sr (float, optional): Enables automatic resampling of audio
            items to the model's expected sampling rate.
        target_channels (int, optional): Target number of audio channels.
            If provided, normalizes audio to this many channels (e.g., 1 for mono).
            If None, audio channels are passed through unchanged.
        expected_hidden_size (int, optional): Expected hidden dimension for
            embedding inputs. If provided, validates that user-supplied
            embeddings have the correct hidden size to prevent crashes
            during model inference.
        if data is None or self._is_empty(data):
            return None
        if self.is_embeddings(data):
            raise ValueError("Do not support embedding data for vision_chunk right now")
        return VisionChunkProcessorItems(data)

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
            "vision_chunk": self._parse_vision_chunk_data,
        }

    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        subparsers = self._get_subparsers()

        mm_items = MultiModalDataItems()
        for k, v in mm_data.items():
            if k not in subparsers:
                raise ValueError(f"Unsupported modality: {k}")

            # ignore empty embedding data
            if (parsed_data := subparsers[k](v)) is not None:
                mm_items[k] = parsed_data

        return mm_items
