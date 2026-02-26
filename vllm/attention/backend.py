# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, get_args

import numpy as np
import torch
from typing_extensions import deprecated

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.attention.backends.utils import KVCacheLayoutType
    from vllm.kv_cache_interface import AttentionSpec

class AttentionType(str, Enum):

    DECODER = "decoder"
    ENCODER_ONLY = "encoder_only"

class MultipleOf:
    base: int

    def __init__(self, base: int):
        self.base = base

class AttentionBackend(ABC):
        Get the physical (memory layout) ordering of the kv cache dimensions.
        e.g. if the KV cache shape is
        [2, num_blocks, block_size, num_heads, head_size],
        and get_kv_cache_stride_order returns (1, 3, 0, 2, 4) then the physical
        ordering of dimensions is
        [num_blocks, num_heads, 2, block_size, head_size].

        If this function is unimplemented / raises NotImplementedError,
        the physical layout of the KV cache will match the logical shape.

        Args:
            include_num_layers_dimension: if True, includes an additional
                num_layers dimension, which is assumed to be prepended
                to the logical KV cache shape.
                With the above example, a return value (2, 4, 0, 1, 3, 5)
                corresponds to
                [num_blocks, num_heads, num_layers, 2, block_size, head_size].

                If an additional dimension is NOT included in the returned
                tuple, the physical layout will not include a layers dimension.

        Returns:
            A tuple of ints which is a permutation of range(len(shape)).

        By default, only supports decoder attention.
        Backends should override this to support other attention types.
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.

    seq_lens: torch.Tensor
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
    num_actual_tokens: int
    max_seq_len: int

    # WARNING: Deprecated fields. Will be removed in a future release (v0.15.0)
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    _num_computed_tokens_cache: torch.Tensor | None = None

    def batch_size(self) -> int:
        return self.seq_lens.shape[0]

    def naive_query_lens(self) -> torch.Tensor:
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release (v0.15.0)
    Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
    async scheduling. If a CPU copy is needed, it can be derived from 
    query_start_loc_cpu and seq_lens.
    Will be removed in a future release (v0.15.0)
        if self._num_computed_tokens_cache is None:
            query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
            self._num_computed_tokens_cache = self.seq_lens - query_lens
        return self._num_computed_tokens_cache

    # TODO(lucas): remove once we have FULL-CG spec-decode support
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            _seq_lens_cpu=self._seq_lens_cpu[:num_actual_reqs]
            if self._seq_lens_cpu is not None
            else None,
            _num_computed_tokens_cpu=self._num_computed_tokens_cpu[:num_actual_reqs]
            if self._num_computed_tokens_cpu is not None
            else None,
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            encoder_seq_lens=maybe_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=maybe_slice_reqs(self.encoder_seq_lens_cpu),
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
        )

M = TypeVar("M")

class AttentionCGSupport(Enum):

    ALWAYS = 3
    the same, this can be used for spec-decode
    NEVER = 0
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    1 + speculative_config.num_speculative_tokens,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        raise NotImplementedError

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        return False

class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...

class AttentionImplBase(ABC, Generic[T]):

    # Required attributes that all impls should have
    num_heads: int
    head_size: int
    scale: float

    # Whether the attention impl can return the softmax lse for decode.
    # Some features like decode context parallelism require the softmax lse.
    can_return_lse_for_decode: bool = False

    # Whether the attention impl supports Prefill Context Parallelism.
    supports_pcp: bool = False
    # Whether the attention impl(or ops) supports MTP
    # when cp_kv_cache_interleave_size > 1
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # some attention backends might not always want to return lse
    # even if they can return lse (for efficiency reasons)
    need_to_return_lse_for_decode: bool = False

    # Whether this attention implementation supports pre-quantized query input.
    # When True, the attention layer will quantize queries before passing them
    # to this backend, allowing torch.compile to fuse the quantization with
    # previous operations. This is typically supported when using FP8 KV cache
    # with compatible attention kernels (e.g., TRT-LLM).
    # Subclasses should set this in __init__.
    # TODO add support to more backends:
    # https://github.com/vllm-project/vllm/issues/25584
    supports_quant_query_input: bool = False
    supports_per_head_quant_scales: bool = False

    dcp_world_size: int
    dcp_rank: int

    pcp_world_size: int
    pcp_rank: int

    total_cp_world_size: int
    total_cp_rank: int

    def __new__(cls, *args, **kwargs):
        # use __new__ so that all subclasses will call this
        self = super().__new__(cls)
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.pcp_world_size = 1
        self.pcp_rank = 0
        self.total_cp_world_size = 1
        self.total_cp_rank = 0
        self.need_to_return_lse_for_decode = False
        return self

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

class AttentionImpl(AttentionImplBase[T], Generic[T]):
        Does this attention implementation support fused output quantization.
        This is used by the AttnFusionPass to only fuse output quantization
        onto implementations that support it.

        :param quant_key: QuantKey object that describes the quantization op
        :return: is fusion supported for this type of quantization

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError

class SparseMLAAttentionImpl(AttentionImplBase[T], Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
