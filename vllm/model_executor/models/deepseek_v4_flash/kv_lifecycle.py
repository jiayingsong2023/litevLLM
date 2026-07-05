# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from .model import DeepSeekV4FlashForCausalLM


class DeepSeekKVLifecycleAdapter:
    """KV lifecycle bridge for DeepSeek model-local paged cache."""

    def __init__(
        self,
        *,
        model: DeepSeekV4FlashForCausalLM,
        context_length: int,
        device: torch.device,
        max_active_requests: int,
    ) -> None:
        self.model = model
        self.context_length = int(context_length)
        self.device = torch.device(device)
        self.max_active_requests = int(max_active_requests)

    @property
    def block_size(self) -> int:
        return self.model.raw_block_size()

    @property
    def num_blocks_per_seq(self) -> int:
        return self.model.num_raw_blocks_per_seq()

    @property
    def num_layers(self) -> int:
        return self.model.num_layers()

    def _ensure_request_state(self, request_id: str) -> None:
        self.model.ensure_request_state(
            request_id=request_id,
            context_length=self.context_length,
            device=self.device,
            max_active_requests=self.max_active_requests,
        )

    def ensure_blocks_for_requests(
        self,
        request_ids: list[str],
        token_counts: list[int],
    ) -> None:
        for request_id, total_tokens in zip(request_ids, token_counts, strict=True):
            self._ensure_request_state(request_id)
            self.model.ensure_request_capacity(
                request_id,
                max(0, int(total_tokens) - 1),
            )

    def free_request_blocks(self, request_id: str) -> None:
        self.model.free_request_state(request_id)

    def stats(self) -> dict[str, Any]:
        return self.model.kv_stats()
