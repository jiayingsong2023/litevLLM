# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.core.block_pool import BlockPool
from vllm.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.request import Request

class SingleTypeKVCacheManager(ABC):
    __slots__ = (
        "block_size",
        "dcp_world_size",
        "pcp_world_size",
        "kv_cache_spec",
        "block_pool",
        "enable_caching",
        "req_to_blocks",
        "num_cached_block",
        "kv_cache_group_id",
        "_null_block",
    )

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:

        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        # The default behavior is to not skip any tokens.
        return 0

class FullAttentionManager(SingleTypeKVCacheManager):
    __slots__ = ()
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

class SlidingWindowManager(SingleTypeKVCacheManager):
    __slots__ = ("sliding_window",)
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length is not compatible with eagle now"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    __slots__ = ("attention_chunk_size",)
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

class MambaManager(SingleTypeKVCacheManager):
    __slots__ = (
        "mamba_cache_mode",
        "num_speculative_blocks",
        "last_state_block_idx",
        "_allocated_block_reqs",
    )

    def __init__(self, kv_cache_spec: MambaSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        super().remove_skipped_blocks(request_id, num_computed_tokens)
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(num_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            if num_new_blocks > 0:
                if request_id in self._allocated_block_reqs:
                    # Old request. Needs at most 1 more blocks as we can reuse the
                    # speculative blocks in previous step.
                    num_new_blocks = 1
                else:
                    # First prefill. Allocate 1 block for running state and the
                    # speculative blocks.
                    num_new_blocks = 1 + self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    "num_required_blocks "
                    f"{num_required_blocks} < len(req_blocks) {len(req_blocks)}"
                )
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null blocks
                if prev_block_len < num_skipped_blocks:
                    req_blocks.extend(
                        [
                            self._null_block
                            for _ in range(prev_block_len, num_skipped_blocks)
                        ]
                    )

                if blocks_allocated:
                    # reuse previous speculative blocks in this step
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            req_blocks.append(req_blocks[block_idx])
                            req_blocks[block_idx] = self._null_block
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                return req_blocks[prev_block_len:]

    def free(self, request_id: str) -> None:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
        super().free(request_id)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return num_computed_tokens - 1

class CrossAttentionManager(SingleTypeKVCacheManager):
