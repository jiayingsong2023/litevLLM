# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)

class CudagraphDispatcher:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE

    def _compute_bs_to_padded_graph_size(self) -> None:
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            disable_full: If True, skip FULL cudagraph checks and
                return PIECEWISE or NONE only. (can be used for features like
                cascade attention that are not supported by full cudagraphs)
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
