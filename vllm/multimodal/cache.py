# SPDX-License-Identifier: Apache-2.0

import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast, Any

import torch
from typing_extensions import override

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.cache import CacheInfo, LRUCache
from vllm.utils.jsontree import json_count_leaves, json_map_leaves, json_reduce_leaves
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.utils.mem_utils import format_gib

from .inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from .processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)

class MultiModalProcessorCacheItem:
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()
        self.item = item
        self.prompt_updates = prompt_updates

class MultiModalProcessorCacheItemMetadata:
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()
        self.item_size = MultiModalCache.get_item_size(item)
        self.prompt_updates = prompt_updates

MultiModalCacheValue: TypeAlias = (
    MultiModalProcessorCacheItem
    | MultiModalProcessorCacheItemMetadata
    | MultiModalKwargsItems
    | MultiModalKwargsItem
    | Mapping[str, NestedTensors]
)

_V = TypeVar("_V", bound=MultiModalCacheValue)

class MultiModalCache:
    @classmethod
    def get_leaf_size(cls, leaf: object) -> int:
        if isinstance(leaf, MultiModalProcessorCacheItem):
            return cls.get_leaf_size(leaf.item)
        if isinstance(leaf, MultiModalProcessorCacheItemMetadata):
            return leaf.item_size
        if isinstance(leaf, (MultiModalKwargsItems, MultiModalKwargsItem, MultiModalFieldElem)):
            return cls.get_item_size(leaf.data)  # type: ignore
        if isinstance(leaf, torch.Tensor):
            return leaf.nbytes
        return sys.getsizeof(leaf)

    @classmethod
    def get_item_size(cls, value: MultiModalCacheValue, *, debug: bool = False) -> int:
        size = json_reduce_leaves(operator.add, json_map_leaves(cls.get_leaf_size, value))
        if debug:
            logger.debug("Calculated size of %s to be %s GiB", type(value), format_gib(size))
        return size

    @classmethod
    def get_item_complexity(cls, value: MultiModalCacheValue) -> int:
        return json_count_leaves(value)

    @classmethod
    def get_lru_cache(cls, capacity_gb: float, value_type: type[_V], *, debug: bool = False) -> LRUCache[str, _V]:
        return LRUCache(GiB_bytes * capacity_gb, getsizeof=lambda x: cls.get_item_size(x, debug=debug))

_I = TypeVar("_I", contravariant=True)
_O = TypeVar("_O", covariant=True)

class BaseMultiModalCache(ABC, Generic[_I, _O]):
    @abstractmethod
    def get_and_update_item(self, mm_item: _I, mm_hash: str) -> _O:
        raise NotImplementedError

    def get_and_update(self, mm_items: Sequence[_I], mm_hashes: list[str]) -> list[_O]:
        assert len(mm_items) == len(mm_hashes)
        return [self.get_and_update_item(mm_item, mm_hash) for mm_item, mm_hash in zip(mm_items, mm_hashes)]

    @abstractmethod
    def clear_cache(self) -> None:
        raise NotImplementedError

MultiModalProcessorCacheInItem: TypeAlias = tuple[MultiModalKwargsItem, Sequence["ResolvedPromptUpdate"]] | None
MultiModalProcessorCacheOutItem: TypeAlias = tuple[MultiModalKwargsItem | None, Sequence["ResolvedPromptUpdate"]]

class BaseMultiModalProcessorCache(BaseMultiModalCache[MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem]):
    @abstractmethod
    def is_cached_item(self, mm_hash: str) -> bool:
        raise NotImplementedError

    def is_cached(self, mm_hashes: list[str]) -> list[bool]:
        return [self.is_cached_item(mm_hash) for mm_hash in mm_hashes]

    def close(self) -> None:
        pass

    @abstractmethod
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        raise NotImplementedError

class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()
        mm_config = model_config.get_multimodal_config()
        self._cache = MultiModalCache.get_lru_cache(mm_config.mm_processor_cache_gb, MultiModalProcessorCacheItem)

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(self, mm_item: MultiModalProcessorCacheInItem, mm_hash: str) -> MultiModalProcessorCacheOutItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item.item, cached_item.prompt_updates
        assert mm_item is not None
        self._cache[mm_hash] = MultiModalProcessorCacheItem(*mm_item)
        return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)

# Alias for litevLLM (no IPC needed)
MultiModalProcessorSenderCache = MultiModalProcessorOnlyCache

class BaseMultiModalReceiverCache(BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem]):
    def get_and_update_features(self, mm_features: list["MultiModalFeatureSpec"]) -> list["MultiModalFeatureSpec"]:
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            self.touch_receiver_cache_item(cache_key, feature.data)
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            feature.data = self.get_and_update_item(feature.data, cache_key)
        return mm_features

    @abstractmethod
    def touch_receiver_cache_item(self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None) -> None:
        raise NotImplementedError

class MultiModalReceiverCache(BaseMultiModalReceiverCache):
    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()
        mm_config = model_config.get_multimodal_config()
        self._cache = MultiModalCache.get_lru_cache(mm_config.mm_processor_cache_gb, MultiModalKwargsItem)

    @override
    def get_and_update_item(self, mm_item: MultiModalKwargsItem | None, mm_hash: str) -> MultiModalKwargsItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item
        assert mm_item is not None
        self._cache[mm_hash] = mm_item
        return mm_item

    @override
    def touch_receiver_cache_item(self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

# Stubs for compatibility
class ShmObjectStoreSenderCache(MultiModalProcessorOnlyCache):
    def __init__(self, vllm_config: "VllmConfig") -> None:
        super().__init__(vllm_config.model_config)

class ShmObjectStoreReceiverCache(MultiModalReceiverCache):
    def __init__(self, vllm_config: "VllmConfig", shared_worker_lock: Any) -> None:
        super().__init__(vllm_config.model_config)
