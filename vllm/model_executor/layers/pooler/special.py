# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Set
from itertools import groupby

import torch

from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.pool.metadata import PoolingMetadata

from .abstract import Pooler, PoolerOutput
from .common import ClassifierFn
from .seqwise import (
    SequencePoolingFn,
    SequencePoolingMethod,
    pooler_for_classify,
    pooler_for_embed,
)
from .tokwise import AllPool, pooler_for_token_classify, pooler_for_token_embed

class DispatchPooler(Pooler):

    def __init__(
        self,
        pooler: Pooler,
        bos_token_id: int = -1,  # -1 disables the filtering
        eos_token_id: int = -1,
    ) -> None:
        super().__init__()

        self.pooler = pooler
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooler.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_outputs = self.pooler(hidden_states, pooling_metadata)
        assert isinstance(pooled_outputs, list)

        for i, prompt_len in enumerate(pooling_metadata.prompt_lens):
            pooled_data = pooled_outputs[i]
            assert (
                isinstance(pooled_data, torch.Tensor)
                and pooled_data.shape[0] == prompt_len
            )
            token_ids = pooling_metadata.prompt_token_ids[i, :prompt_len]
            if token_ids[0] == self.bos_token_id:
                pooled_data = pooled_data[1:]
            if token_ids[-1] == self.eos_token_id:
                pooled_data = pooled_data[:-1]
            pooled_outputs[i] = pooled_data.squeeze()

        return pooled_outputs

__all__ = ["BOSEOSFilter", "DispatchPooler", "IdentityPooler"]
