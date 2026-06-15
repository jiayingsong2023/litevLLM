# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
)


def _payload_bytes(payload: torch.Tensor) -> bytes:
    if payload.dtype != torch.uint8:
        raise ValueError(f"payload must be torch.uint8; got {payload.dtype}")
    if not payload.is_cuda:
        raise ValueError("payload must be a CUDA tensor")
    return bytes(payload.detach().cpu().contiguous().tolist())


def _validate_hidden(hidden: torch.Tensor, *, columns: int) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError("hidden must be a CUDA tensor")
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if hidden.numel() != columns:
        raise ValueError(
            "hidden length must match matrix columns; "
            f"got hidden={hidden.numel()} and columns={columns}"
        )
    return hidden.to(torch.float32)


def deepseek_v4_q2_k_matvec(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Correctness-first Q2_K expert matvec fallback.

    GGUF raw Q2_K block memory layout:
    - One block encodes 256 row-major matrix values in 84 bytes.
    - Bytes 0..15 hold 16 packed scale/min pairs, with low nibble as scale
      and high nibble as minimum scale.
    - Bytes 16..79 hold 64 bytes of 2-bit quantized values.
    - Bytes 80..81 hold the fp16 block scale d, and bytes 82..83 hold the
      fp16 minimum scale dmin.

    Intended Triton tiling:
    - Treat the matrix as [rows, columns] and split each row into 256-value
      GGUF blocks.
    - A fused kernel should tile output rows and K blocks, decode Q2_K values
      in registers, multiply by hidden[K], and reduce to one fp32 value per row.

    This implementation is deliberately not the final kernel. It copies the
    staged CUDA uint8 payload to CPU bytes, decodes with the reference GGUF
    decoder, moves the fp32 matrix back to CUDA, and performs torch matmul.
    """
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    matrix = q2_k_matrix_from_gguf_payload(
        _payload_bytes(payload),
        rows=rows,
        columns=columns,
    ).to(device=hidden.device)
    return matrix.matmul(hidden_f32)


def deepseek_v4_iq2_xxs_matvec(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Correctness-first IQ2_XXS expert matvec fallback.

    GGUF raw IQ2_XXS block memory layout:
    - One block encodes 256 row-major matrix values in 66 bytes.
    - Bytes 0..1 hold the fp16 block scale d.
    - Bytes 2..65 hold eight 8-byte groups. Each group carries packed grid
      indices and sign/scale metadata that the GGML IQ2_XXS reference expands
      into eight lanes of 2-bit grid values and signs.

    Intended Triton tiling:
    - Treat the matrix as [rows, columns] and split each row into 256-value
      GGUF blocks.
    - A fused kernel should tile output rows and K blocks, expand IQ2_XXS grid
      and sign metadata in registers, multiply by hidden[K], and reduce to one
      fp32 value per row.

    This implementation is deliberately not the final kernel. It copies the
    staged CUDA uint8 payload to CPU bytes, decodes with the reference GGUF
    decoder, moves the fp32 matrix back to CUDA, and performs torch matmul.
    """
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    matrix = iq2_xxs_matrix_from_gguf_payload(
        _payload_bytes(payload),
        rows=rows,
        columns=columns,
    ).to(device=hidden.device)
    return matrix.matmul(hidden_f32)
