# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.platform_utils import is_pin_memory_available

pin_memory = is_pin_memory_available()

@dataclass
class PoolingCursor:
    index: list[int]
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
        self.hidden_states_cache: list[torch.Tensor] = []

    def clean(self):
        self.hidden_states_cache.clear()

@dataclass
class PoolingMetadata:
