# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from .config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from .gguf_reader import GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0
from .quant import q8_0_matvec
from .weight_store import DeepSeekV4FlashWeightStore

_RMS_NORM_EPS = 1e-6
_Q8_0_BLOCK_SIZE = 32
_Q8_0_BLOCK_BYTES = 2 + _Q8_0_BLOCK_SIZE
_OUTPUT_PROJECTION_CHUNK_ROWS = 1024
_READONLY_BUFFER_WARNING = "The given buffer is not writable"


class DeepSeekV4FlashForCausalLM(nn.Module):
    """DeepSeek V4 Flash GGUF-backed bring-up model.

    The current forward is intentionally limited to a real tensor smoke path:
    token embedding, final RMSNorm, and output projection. It does not execute
    transformer layers yet.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE,
        weight_store: DeepSeekV4FlashWeightStore | None = None,
        runtime_budget: DeepSeekV4FlashRuntimeBudget | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.shape = shape
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget
        self.limited_forward_smoke_only = True

    def attach_weight_store(
        self,
        weight_store: DeepSeekV4FlashWeightStore,
        runtime_budget: DeepSeekV4FlashRuntimeBudget,
    ) -> None:
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget

    def close(self) -> None:
        if self.weight_store is not None:
            self.weight_store.close()
            self.weight_store = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        if self.weight_store is None:
            raise RuntimeError(
                "DeepSeekV4FlashForCausalLM requires an attached GGUF weight store"
            )
        if input_ids.ndim != 1:
            raise ValueError(
                "DeepSeekV4FlashForCausalLM.forward only supports 1-D input_ids "
                f"for batch=1 smoke; got {input_ids.ndim}-D"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"input_ids must use an integer dtype; got {input_ids.dtype}"
            )
        if self.runtime_budget is not None:
            context_length = self.runtime_budget.context.context_length
            if context_length > 8192:
                raise ValueError(
                    "DeepSeekV4FlashForCausalLM first release supports context <= "
                    f"8192; got runtime context {context_length}"
                )
        if input_ids.numel() == 0:
            return torch.empty((0, self.shape.vocab_size), dtype=torch.float32)
        if input_ids.numel() != 1:
            raise ValueError(
                "DeepSeekV4FlashForCausalLM limited smoke supports one token until "
                "full transformer forward is implemented; "
                f"got {input_ids.numel()} input tokens"
            )
        return self._forward_embedding_output_projection_smoke_batch_one(input_ids)

    def _forward_embedding_output_projection_smoke_batch_one(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a limited real-tensor projection smoke, not transformer layers."""
        store = self._require_weight_store()
        logits = [
            self._project_one_token_embedding_smoke(store, int(token_id))
            for token_id in input_ids.detach().cpu().tolist()
        ]
        return torch.stack(logits, dim=0)

    def _require_weight_store(self) -> DeepSeekV4FlashWeightStore:
        if self.weight_store is None:
            raise RuntimeError(
                "DeepSeekV4FlashForCausalLM requires an attached GGUF weight store"
            )
        return self.weight_store

    def _project_one_token_embedding_smoke(
        self,
        store: DeepSeekV4FlashWeightStore,
        token_id: int,
    ) -> torch.Tensor:
        hidden = self._read_token_embedding(store, token_id)
        norm_weight = self._read_output_norm(store)
        normalized = self._rms_norm(hidden, norm_weight)
        return self._q8_0_output_projection(store, normalized)

    def _read_token_embedding(
        self,
        store: DeepSeekV4FlashWeightStore,
        token_id: int,
    ) -> torch.Tensor:
        tensor = store.bindings.token_embedding
        if tensor.tensor_type != GGML_TYPE_F16:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects F16 token_embd.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        hidden_size = self.shape.hidden_size
        vocab_size = self.shape.vocab_size
        if tensor.dims != (hidden_size, vocab_size):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects token_embd.weight dims "
                f"({hidden_size}, {vocab_size}); got {tensor.dims}"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"input token id {token_id} is outside vocab range [0, {vocab_size})"
            )

        row_bytes = hidden_size * 2
        row_start = token_id * row_bytes
        payload = store.tensor_payload(tensor)
        row_view = payload[row_start : row_start + row_bytes]
        try:
            return _torch_from_readonly_buffer(row_view, dtype=torch.float16).to(
                torch.float32
            )
        finally:
            row_view.release()
            payload.release()

    def _read_output_norm(self, store: DeepSeekV4FlashWeightStore) -> torch.Tensor:
        tensor = store.bindings.output_norm
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash smoke requires output_norm.weight")
        if tensor.tensor_type != GGML_TYPE_F32:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects F32 output_norm.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        if tensor.dims != (self.shape.hidden_size,):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects output_norm.weight dims "
                f"({self.shape.hidden_size},); got {tensor.dims}"
            )
        payload = store.tensor_payload(tensor)
        try:
            return _torch_from_readonly_buffer(payload, dtype=torch.float32).clone()
        finally:
            payload.release()

    def _rms_norm(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = hidden.to(torch.float32).pow(2).mean()
        scale = torch.rsqrt(variance + _RMS_NORM_EPS)
        return hidden.to(torch.float32) * scale * weight.to(torch.float32)

    def _q8_0_output_projection(
        self,
        store: DeepSeekV4FlashWeightStore,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        tensor = store.bindings.output_head
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash smoke requires output.weight")
        if tensor.tensor_type != GGML_TYPE_Q8_0:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects Q8_0 output.weight; "
                f"got GGML type {tensor.tensor_type}"
            )

        columns = self.shape.hidden_size
        rows = self.shape.vocab_size
        if tensor.dims != (columns, rows):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects output.weight dims "
                f"({columns}, {rows}); got {tensor.dims}"
            )
        if columns % _Q8_0_BLOCK_SIZE != 0:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke requires hidden size divisible by "
                f"{_Q8_0_BLOCK_SIZE}; got {columns}"
            )

        blocks_per_row = columns // _Q8_0_BLOCK_SIZE
        row_bytes = blocks_per_row * _Q8_0_BLOCK_BYTES
        logits = torch.empty(rows, dtype=torch.float32)
        payload = store.tensor_payload(tensor)
        try:
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                chunk_rows = row_end - row_start
                byte_start = row_start * row_bytes
                byte_end = row_end * row_bytes
                chunk_view = payload[byte_start:byte_end]
                try:
                    values, scales = self._decode_q8_0_output_chunk(
                        chunk_view,
                        rows=chunk_rows,
                        columns=columns,
                        blocks_per_row=blocks_per_row,
                    )
                    logits[row_start:row_end] = q8_0_matvec(
                        values,
                        scales,
                        hidden,
                        block_size=_Q8_0_BLOCK_SIZE,
                    )
                finally:
                    chunk_view.release()
        finally:
            payload.release()
        return logits

    def _decode_q8_0_output_chunk(
        self,
        payload: memoryview,
        *,
        rows: int,
        columns: int,
        blocks_per_row: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = _torch_from_readonly_buffer(payload, dtype=torch.uint8).reshape(
            rows * blocks_per_row,
            _Q8_0_BLOCK_BYTES,
        )
        scale_bytes = raw[:, :2].contiguous()
        values = raw[:, 2:].contiguous().view(torch.int8).reshape(rows, columns)
        scales = scale_bytes.view(torch.float16).to(torch.float32).reshape(
            rows,
            blocks_per_row,
        )
        return values, scales


def _torch_from_readonly_buffer(
    payload: memoryview,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_READONLY_BUFFER_WARNING)
        return torch.frombuffer(payload, dtype=dtype)
