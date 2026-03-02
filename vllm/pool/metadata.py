# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

@dataclass
class PoolingCursor:
    index: List[int]
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor

    def __getitem__(self, indices: slice):
        return PoolingCursor(
            index=self.index[indices],
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            seq_lens_cpu=self.seq_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self):
        return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)

    def is_finished(self):
        return self.prompt_lens_cpu == self.seq_lens_cpu

class PoolingStates:
    def __init__(self):
        # for chunked prefill with ALL pooling
        self.hidden_states_cache: List[torch.Tensor] = []

    def clean(self):
        self.hidden_states_cache.clear()

@dataclass
class PoolingMetadata:
    """Metadata for pooling operations."""
    seq_groups: List[int]
    seq_data: dict
    pooling_params: PoolingParams
    cursor: Optional[PoolingCursor] = None
    
    def __post_init__(self):
        # Initialization logic if needed
        pass
