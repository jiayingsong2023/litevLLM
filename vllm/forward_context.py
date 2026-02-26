# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.attention.backend import AttentionMetadata
from vllm.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)

class BatchDescriptor(NamedTuple):

    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )

def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()

def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    sp_size = len(sp_tokens)

    local_size = [-1] * sp_size
    for i in range(sp_size):
        # Take into account sharding if MoE activation is sequence parallel.
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size

@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        assert parallel_config.is_moe_model is not False
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        self.local_sizes = _compute_chunked_local_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
            max_chunk_size_per_rank,
            chunk_idx,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cummulative num tokens across DP.
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)

@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None

    # If True, bypass the compiled model call, e.g. by using .forward() directly
    skip_compiled: bool = False

    # For torch.compile cold start times, we need to avoid hard-coding
    # any strings into the graph. Right now, the vllm.moe_forward
    # and vllm.moe_forward_shared custom operators hard-code strings into
    # the graph.
    #
    # The workaround is to store a list of the strings that each of those
    # custom ops needs in the ForwardContext (all_moe_layers)
    # as well as a counter (moe_layer_index).
    # The ForwardContext object is alive for the duration of the forward pass.
    # When the custom op needs a layer string, get the next string
    # from all_moe_layers and increment the counter.
    #
    # This assumes that the custom operators will always be executed in
    # order and that torch.compile will not try to reorder these
    # operations with respect to each other.
    #
    # TODO(https://github.com/vllm-project/vllm/issues/31985):
    # There are longer-term solutions, like unwrapping the moe custom operator,
    # that aren't ready yet.
    # We could also treat the string as a "symbolic input" to the graph but
    # the PyTorch-side bits for that aren't ready yet either.
    #
    # If this value is None (like in some tests), then we end up baking the string
    # into the graph. Otherwise, the moe custom ops will pop a string from this list.
    all_moe_layers: list[str] | None = None
    moe_layer_index: int = 0

    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.cudagraph_runtime_mode.valid_runtime_modes(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )

_forward_context: ForwardContext | None = None

def get_forward_context() -> ForwardContext:
    This is used to override the forward context for a specific
    forward pass.
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
