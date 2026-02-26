# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
from itertools import accumulate
from math import prod

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return prod(shape) * dtype.itemsize

# Constants
_MB = 1024**2
_GiB = 1024**3

# Global workspace manager instance
_manager: "WorkspaceManager | None" = None

class WorkspaceManager:

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        self._current_workspaces: list[torch.Tensor | None] = [None, None]
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        return self._locked

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        current_workspace = self._ensure_workspace_size(total_bytes)

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        ubatch_id = dbo_current_ubatch_id()
        current_workspace = self._current_workspaces[ubatch_id]
        current_size = self._workspace_size_bytes(current_workspace)

        if current_size < required_bytes:

            def get_caller_info() -> str:

    Returns:
        True if workspace manager is initialized, False otherwise.

    Raises:
        AssertionError: If workspace manager has not been initialized.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of micro-batches. Defaults to 1.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
