# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypedDict,
    Union,
    cast,
    final,
)

import numpy as np
from PIL.Image import Image
from typing_extensions import NotRequired, TypeVar

from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader
from vllm.utils.jsontree import json_map_leaves

from .media import MediaWithBytes

if TYPE_CHECKING:
    import torch
    import torch.types
    from transformers.feature_extraction_utils import BatchFeature
else:
    torch = LazyLoader("torch", globals(), "torch")

_T = TypeVar("_T")

HfImageItem: TypeAlias = Union["Image", np.ndarray, "torch.Tensor"]

HfVideoItem: TypeAlias = Union[
    list["Image"], np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]
]

HfAudioItem: TypeAlias = Union[list[float], np.ndarray, "torch.Tensor"]

ImageItem: TypeAlias = Union[HfImageItem, "torch.Tensor", MediaWithBytes[HfImageItem]]

VideoItem: TypeAlias = Union[
    HfVideoItem, "torch.Tensor", tuple[HfVideoItem, dict[str, Any]]
]

AudioItem: TypeAlias = Union[HfAudioItem, tuple[np.ndarray, float], "torch.Tensor"]

ModalityData: TypeAlias = _T | list[_T | None] | None

class VisionChunkImage(TypedDict):

    type: Literal["video_chunk"]
    video_chunk: list[Image]
    uuid: str | None
    prompt: str
    video_idx: int

VisionChunk = VisionChunkImage | VisionChunkVideo

    image: ModalityData[ImageItem]

    audio: ModalityData[AudioItem]

MultiModalDataDict: TypeAlias = Mapping[str, ModalityData[Any]]

MultiModalUUIDDict: TypeAlias = Mapping[str, list[str | None] | str]

@dataclass(frozen=True)
class PlaceholderRange:

    offset: int

    is_embed: "torch.Tensor | None" = None

    @cached_property
    def embeds_cumsum(self) -> torch.Tensor | None:
        return None if self.is_embed is None else self.is_embed.cumsum(dim=0)

    @cached_property
    def get_num_embeds(self) -> int:
        if self.embeds_cumsum is None:
            return self.length

        return int(self.embeds_cumsum[-1])

    def get_embeds_indices_in_range(
        self, start_idx: int, end_idx: int
    ) -> tuple[int, int]:
        if self.embeds_cumsum is None:
            return start_idx, end_idx

        embeds_start_idx = (
            int(self.embeds_cumsum[start_idx - 1]) if start_idx > 0 else 0
        )
        embeds_end_idx = int(self.embeds_cumsum[end_idx - 1])

        return embeds_start_idx, embeds_end_idx

    def extract_embeds_range(self) -> list[tuple[int, int]]:
        if self.is_embed is None:
            return [(self.offset, self.offset + self.length - 1)]

        mask_i = self.is_embed.int()
        starts = torch.nonzero(
            torch.diff(mask_i, prepend=mask_i.new_zeros(1)) == 1
        ).flatten()
        ends = torch.nonzero(
            torch.diff(mask_i, append=mask_i.new_zeros(1)) == -1
        ).flatten()
        ranges = torch.stack((starts, ends), dim=1) + self.offset
        return [tuple(x) for x in ranges.tolist()]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if not (self.offset, self.length) == (other.offset, other.length):
            return False

        if self.is_embed is None:
            return other.is_embed is None
        if other.is_embed is None:
            return self.is_embed is None

        return nested_tensors_equal(self.is_embed, other.is_embed)

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]

def nested_tensors_equal(a: NestedTensors, b: NestedTensors) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and torch.equal(b, a)

    if isinstance(a, list):
        return isinstance(b, list) and all(
            nested_tensors_equal(a_, b_) for a_, b_ in zip(a, b)
        )
    if isinstance(b, list):
        return isinstance(a, list) and all(
            nested_tensors_equal(b_, a_) for b_, a_ in zip(b, a)
        )

    # Both a and b are scalars
    return a == b

def _nested_tensors_h2d(
    tensors: NestedTensors,
    device: torch.types.Device,
) -> NestedTensors:
    if device is None:
        return tensors

    return json_map_leaves(
        (
            lambda x: x.to(device=device, non_blocking=True)
            if isinstance(x, torch.Tensor)
            else x
        ),
        tensors,
    )

BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]

def batched_tensors_equal(a: BatchedTensorInputs, b: BatchedTensorInputs) -> bool:
    return all(k in b and nested_tensors_equal(a[k], b[k]) for k in a)

@dataclass
class MultiModalFeatureSpec:

    data: "MultiModalKwargsItem | None"

    modality: str

    mm_position: PlaceholderRange

    mm_hash: str | None = None
    Represents a processed keyword argument to pass to a model for a
    [`MultiModalKwargsItem`][vllm.multimodal.inputs.MultiModalKwargsItem].
    The tensor data of this field in
    [`MultiModalKwargsItem`][vllm.multimodal.inputs.MultiModalKwargsItem],
    i.e. the value of the keyword argument to be passed to the model.

    It may be set to `None` if it is determined that the item is cached
    in `EngineCore`.
    Defines how to combine the tensor data of this field with others
    in order to batch multi-modal items together for model inference.
    Defines how to interpret tensor data belonging to a keyword argument for
    [`MultiModalKwargsItems`][vllm.multimodal.inputs.MultiModalKwargsItems],
    and vice versa.
    If `True`, then this field is excluded from being moved to the accelerator
    when `MultiModalKwargsItems.get_data()` is called to batch the data.
        Construct
        [`MultiModalFieldElem`][vllm.multimodal.inputs.MultiModalFieldElem]
        instances to represent the provided data.

        This is the inverse of
        [`reduce_data`][vllm.multimodal.inputs.BaseMultiModalField.reduce_data].
        Merge the data from multiple instances of
        [`MultiModalFieldElem`][vllm.multimodal.inputs.MultiModalFieldElem].

        This is the inverse of
        [`build_elems`][vllm.multimodal.inputs.BaseMultiModalField.build_elems].
    Info:
        [`MultiModalFieldConfig.batched`][vllm.multimodal.inputs.MultiModalFieldConfig.batched]
    Info:
        [`MultiModalFieldConfig.flat`][vllm.multimodal.inputs.MultiModalFieldConfig.flat]
        [`MultiModalFieldConfig.flat_from_sizes`][vllm.multimodal.inputs.MultiModalFieldConfig.flat_from_sizes]
    Info:
        [`MultiModalFieldConfig.shared`][vllm.multimodal.inputs.MultiModalFieldConfig.shared]
        Defines a field where an element in the batch is obtained by
        indexing into the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Input:
            Data: [[AAAA]
                [BBBB]
                [CCCC]]

        Output:
            Element 1: [AAAA]
            Element 2: [BBBB]
            Element 3: [CCCC]
        ```
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            slices: For each multi-modal item, a slice (dim=0) or a tuple of
                slices (dim>0) that is used to extract the data corresponding
                to it.
            dim: The dimension to extract data, default to 0.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            slices: [slice(0, 3), slice(3, 7), slice(7, 9)]

        Input:
            Data: [AAABBBBCC]

        Output:
            Element 1: [AAA]
            Element 2: [BBBB]
            Element 3: [CC]
        ```

        ```
        Given:
            slices: [
                (slice(None), slice(0, 3)),
                (slice(None), slice(3, 7)),
                (slice(None), slice(7, 9))]
            dim: 1

        Input:
            Data: [[A],[A],[A],[B],[B],[B],[B],[C],[C]]

        Output:
            Element 1: [[A],[A],[A]]
            Element 2: [[B],[B],[B],[B]]
            Element 3: [[C],[C]]
        ```
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            size_per_item: For each multi-modal item, the size of the slice
                that is used to extract the data corresponding to it.
            dim: The dimension to slice, default to 0.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            size_per_item: [3, 4, 2]

        Input:
            Data: [AAABBBBCC]

        Output:
            Element 1: [AAA]
            Element 2: [BBBB]
            Element 3: [CC]
        ```

        ```
        Given:
            size_per_item: [3, 4, 2]
            dim: 1

        Input:
            Data: [[A],[A],[A],[B],[B],[B],[B],[C],[C]]

        Output:
            Element 1: [[A],[A],[A]]
            Element 2: [[B],[B],[B],[B]]
            Element 3: [[C],[C]]
        ```

        Info:
            [`MultiModalFieldConfig.flat`][vllm.multimodal.inputs.MultiModalFieldConfig.flat]
        Defines a field where an element in the batch is obtained by
        taking the entirety of the underlying data.

        This means that the data is the same for each element in the batch.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            batch_size: The number of multi-modal items which share this data.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            batch_size: 4

        Input:
            Data: [XYZ]

        Output:
            Element 1: [XYZ]
            Element 2: [XYZ]
            Element 3: [XYZ]
            Element 4: [XYZ]
        ```
    A dictionary of processed keyword arguments to pass to the model,
    corresponding to a single item in
    [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems].
        mm_elem = MultiModalFieldElem(
            data=torch.empty(nbytes, dtype=torch.uint8),
            field=MultiModalSharedField(batch_size=1),
        )
        return MultiModalKwargsItem({"dummy": mm_elem})

    def get_data(self) -> dict[str, NestedTensors]:
        return {key: elem.data for key, elem in self.items()}

_I = TypeVar(
    "_I",
    MultiModalKwargsItem,
    MultiModalKwargsItem | None,
    default=MultiModalKwargsItem,
)

class MultiModalKwargsItems(UserDict[str, Sequence[_I]]):

    @staticmethod
    def from_hf_inputs(
        hf_inputs: "BatchFeature",
        config_by_key: Mapping[str, MultiModalFieldConfig],
    ):
        # NOTE: This skips fields in `hf_inputs` that are not in `config_by_key`
        # We assume that those fields are not used in vLLM
        elems_by_key = dict[str, Sequence[MultiModalFieldElem]]()
        keys_by_modality = defaultdict[str, set[str]](set)
        for key, config in config_by_key.items():
            batch = hf_inputs.get(key)
            if batch is not None:
                elems = config.build_elems(key, batch)
                if len(elems) > 0:
                    elems_by_key[key] = elems
                    keys_by_modality[config.modality].add(key)

        items_by_modality = dict[str, list[MultiModalKwargsItem]]()
        for modality, keys in keys_by_modality.items():
            elems_in_modality = {k: elems_by_key[k] for k in keys}
            batch_sizes = {k: len(v) for k, v in elems_in_modality.items()}

            if len(set(batch_sizes.values())) > 1:
                raise ValueError(
                    f"Cannot merge different batch sizes for {modality=}! "
                    f"Found: {batch_sizes=}"
                )

            batch_size = next(iter(batch_sizes.values()))
            items_by_modality[modality] = [
                MultiModalKwargsItem({k: v[i] for k, v in elems_in_modality.items()})
                for i in range(batch_size)
            ]

        return MultiModalKwargsItems(items_by_modality)

    def __getitem__(self, modality: str) -> Sequence[_I]:
        if modality not in self:
            raise KeyError(
                f"Modality {modality!r} not found. "
                f"Available modalities: {set(self.keys())}"
            )

        return super().__getitem__(modality)  # type: ignore[return-value]

    def require_data(self) -> "MultiModalKwargsItems[MultiModalKwargsItem]":
        for modality, items in self.items():
            for i, item in enumerate(items):
                if item is None:
                    raise RuntimeError(f"Found empty mm_items[{modality}][{i}]")

        return self  # type: ignore[return-value]

    def get_data(
        self,
        *,
        device: torch.types.Device = None,
        pin_memory: bool = False,
    ) -> BatchedTensorInputs:
A dictionary containing per-item hashes for each modality.
A dictionary containing per-item placeholder ranges for each modality.
    Represents the outputs of
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor],
    ready to be passed to vLLM internals.

    prompt_token_ids: list[int]

    mm_hashes: MultiModalHashes
    For each modality, information about the placeholder tokens in
    `prompt_token_ids`.
    Optional cache salt to be used for prefix caching.
    Represents the outputs of
    [`EncDecMultiModalProcessor`][vllm.multimodal.processing.EncDecMultiModalProcessor]
    ready to be passed to vLLM internals.
