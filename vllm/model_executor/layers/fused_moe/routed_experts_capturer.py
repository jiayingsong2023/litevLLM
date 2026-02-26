# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

logger = logging.getLogger(__name__)

# Constants
_TMP_DIR = tempfile.gettempdir()
_LOCK_FILE_PREFIX = os.path.join(_TMP_DIR, "vllm_routed_experts")
_BUFFER_PREFIX = "vllm_routed_experts_buffer"

# Global singleton instances
_global_experts_capturer: RoutedExpertsCapturer | None = None
_global_experts_reader: RoutedExpertsReader | None = None

@contextmanager
def _file_lock(lock_file: str, mode: str = "wb+") -> Generator[None, None, None]:
    # Ensure lock file exists before acquiring lock
    with open(lock_file, "wb"):
        pass

    with _file_lock(lock_file):
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name, create=False, size=size)

        if shm.size != size:
            logger.warning(
                "Shared memory %s size mismatch; recreating",
                name,
            )
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                logger.info("Created shared memory %s", name)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False, size=size)
                logger.info("Linked to existing shared memory %s", name)

    return shm

class RoutedExpertsCapturer:

    _instance: RoutedExpertsCapturer | None = None

    def __init__(self) -> None:
        self._device_buffer: torch.Tensor | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._lock_file: str | None = None

    @classmethod
    def create(cls) -> RoutedExpertsCapturer:
        return _global_experts_capturer

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:

        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok

        # Initialize device buffer
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device="cuda",
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return

        # Initialize shared memory
        shape = (max_num_kv_tokens, num_layers, num_experts_per_tok)
        buffer_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize
        instance_id = vllm_config.instance_id
        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"

        self._shm = _create_or_attach_shared_memory(
            shm_name, buffer_size, self._lock_file
        )
        self._host_buffer_view = np.ndarray(shape, dtype=np.int32, buffer=self._shm.buf)
        self._host_buffer_view.fill(0)

        logger.debug(
            "Created shared memory buffer '%s' with shape %s",
            shm_name,
            shape,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            token_num_per_dp = ctx.dp_metadata.num_tokens_across_dp_cpu[self.dp_rank]
            cumsum = torch.cumsum(ctx.dp_metadata.num_tokens_across_dp_cpu, dim=0)
            assert cumsum[-1] == topk_ids.shape[0]
            end_loc = cumsum[self.dp_rank]
            start_loc = end_loc - token_num_per_dp

        if layer_id >= self._device_buffer.shape[1]:
            return

        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        Save captured experts from device buffer to shared memory.

        Args:
            indices: Array of indices indicating where to store the data.
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                logger.debug("Exception during cleanup for capturer", exc_info=True)
            finally:
                self._shm = None

    def __del__(self) -> None:
    Reader for routed experts from shared memory.

    This class attaches to shared memory created by RoutedExpertsCapturer
    and reads expert routing decisions.
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts reader already created.")

        _global_experts_reader = cls()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> RoutedExpertsReader | None:
        Attach to an existing shared memory buffer.

        Args:
            max_num_kv_tokens: Maximum number of KV tokens.
            vllm_config: vllm configuration.
        Read routed expert data from shared memory.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the expert routing data for the given indices.
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                logger.debug("Exception during cleanup for reader", exc_info=True)
            finally:
                self._shm = None

    def __del__(self) -> None:
