# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from typing import Any

import torch

from vllm.engine.executor_result import TokenDecodeResult, TokenPrefillResult


class DeepSeekPrefillExecutor:
    def __init__(self, *, model: Any, observer: Any | None) -> None:
        self.model = model
        self.observer = observer

    def execute(
        self,
        request_ids: list[str],
        scheduler: Any,
        chunk_len: int,
    ) -> TokenPrefillResult:
        del chunk_len
        next_tokens: list[int] = []
        prefilled_tokens: list[int] = []
        for request_id in request_ids:
            request = scheduler.get_request(request_id)
            max_tokens = int(request.sampling_params.max_tokens or 1)
            token = self.model.prefill_request(
                request_id,
                list(request.input_ids),
                max_tokens,
            )
            next_tokens.append(int(token))
            prefilled_tokens.append(len(request.input_ids))
        return TokenPrefillResult(
            next_token_ids=torch.tensor(
                next_tokens,
                dtype=torch.long,
                device=self.model.device(),
            ),
            prefilled_tokens=prefilled_tokens,
            is_last_chunk=[True] * len(request_ids),
        )


class DeepSeekDecodeExecutor:
    def __init__(self, *, model: Any, observer: Any | None) -> None:
        self.model = model
        self.observer = observer

    def execute_sync_fast(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> TokenDecodeResult:
        return self.execute_batch(request_ids, scheduler)

    def execute_batch(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> TokenDecodeResult:
        del scheduler
        start = time.perf_counter()
        decode_batch = getattr(self.model, "decode_tokens_batch", None)
        if callable(decode_batch):
            next_token_ids = decode_batch(request_ids).to(
                device=self.model.device(),
                dtype=torch.long,
            )
        else:
            next_token_ids = torch.tensor(
                [
                    int(self.model.decode_single_token(request_id))
                    for request_id in request_ids
                ],
                dtype=torch.long,
                device=self.model.device(),
            )
        record = getattr(self.observer, "on_deepseek_event", None)
        if callable(record):
            record(
                "decode_batch",
                batch_size=len(request_ids),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        return TokenDecodeResult(next_token_ids=next_token_ids)
