# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set

import torch
import torch.nn as nn

from vllm.tasks import PoolingTask
from vllm.v1_outputs import PoolerOutput
from vllm.pool.metadata import PoolingMetadata

from .common import PoolingParamsUpdate

class Pooler(nn.Module, ABC):
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError

__all__ = ["Pooler"]
