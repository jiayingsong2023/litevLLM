# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable

import pytest
import torch

_ROWS = 1024
_COLUMNS = 256
_WARMUP = 3
_REPEAT = 10


def _iq2_xxs_deterministic_payload(rows: int) -> bytes:
    blocks: list[bytes] = []
    for row in range(rows):
        groups = bytearray()
        for group in range(8):
            grid_bytes = bytes(
                ((row * 19 + group * 29 + idx * 37 + 0x17) & 0xFF) for idx in range(4)
            )
            sign_indices = [
                (row * 11 + group * 7 + idx * 23 + 0x05) & 0x7F for idx in range(4)
            ]
            scale_code = (row + group * 3) & 0x0F
            q_sign_scale = (
                sign_indices[0]
                | (sign_indices[1] << 7)
                | (sign_indices[2] << 14)
                | (sign_indices[3] << 21)
                | (scale_code << 28)
            )
            groups.extend(grid_bytes)
            groups.extend(struct.pack("<I", q_sign_scale))
        d = 0.03125 * (row % 7 + 1)
        blocks.append(struct.pack("<e", d) + bytes(groups))
    return b"".join(blocks)


def _q2_k_deterministic_payload(rows: int) -> bytes:
    blocks: list[bytes] = []
    for row in range(rows):
        scales = bytes(((row * 5 + idx * 7 + 0x21) & 0xFF) for idx in range(16))
        qs = bytes(((row * 13 + idx * 11 + 0x35) & 0xFF) for idx in range(64))
        d = 0.125 * (row % 5 + 1)
        dmin = -0.0625 * (row % 7 + 1)
        blocks.append(scales + qs + struct.pack("<e", d) + struct.pack("<e", dmin))
    return b"".join(blocks)


def _cuda_payload(payload: bytes) -> torch.Tensor:
    return torch.tensor(tuple(payload), dtype=torch.uint8, device="cuda")


def _effective_gbps(
    *,
    payload: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
    elapsed_ms: float,
) -> float:
    if elapsed_ms <= 0.0:
        return 0.0
    total_bytes = (
        payload.numel() * payload.element_size()
        + hidden.numel() * hidden.element_size()
        + output.numel() * output.element_size()
    )
    return total_bytes / (elapsed_ms / 1000.0) / 1_000_000_000.0


def _measure_matvec(
    matvec: Callable[..., torch.Tensor],
    payload: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    output = matvec(payload, hidden, rows=_ROWS, columns=_COLUMNS)
    for _ in range(_WARMUP):
        output = matvec(payload, hidden, rows=_ROWS, columns=_COLUMNS)
    torch.cuda.synchronize()

    elapsed_ms = 0.0
    for _ in range(_REPEAT):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = matvec(payload, hidden, rows=_ROWS, columns=_COLUMNS)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms += float(start_event.elapsed_time(end_event))
    return elapsed_ms / _REPEAT, output


def test_deepseek_q2_iq2_kernel_profile() -> None:
    if os.environ.get("RUN_DEEPSEEK_Q2_IQ2_PROFILE") != "1":
        pytest.skip("set RUN_DEEPSEEK_Q2_IQ2_PROFILE=1 to profile Q2/IQ2 kernels")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.kernels.triton.deepseek_v4_flash.q2_iq2_moe import (
        deepseek_v4_iq2_xxs_matvec,
        deepseek_v4_q2_k_matvec,
    )

    hidden = torch.linspace(-1.0, 1.0, _COLUMNS, dtype=torch.float32, device="cuda")
    q2_payload = _cuda_payload(_q2_k_deterministic_payload(_ROWS))
    iq2_payload = _cuda_payload(_iq2_xxs_deterministic_payload(_ROWS))

    q2_ms, q2_output = _measure_matvec(
        deepseek_v4_q2_k_matvec,
        q2_payload,
        hidden,
    )
    iq2_ms, iq2_output = _measure_matvec(
        deepseek_v4_iq2_xxs_matvec,
        iq2_payload,
        hidden,
    )

    metrics = {
        "iq2_xxs_effective_gbps": _effective_gbps(
            payload=iq2_payload,
            hidden=hidden,
            output=iq2_output,
            elapsed_ms=iq2_ms,
        ),
        "iq2_xxs_ms": iq2_ms,
        "q2_k_effective_gbps": _effective_gbps(
            payload=q2_payload,
            hidden=hidden,
            output=q2_output,
            elapsed_ms=q2_ms,
        ),
        "q2_k_ms": q2_ms,
    }
    print(json.dumps(metrics, sort_keys=True))

    assert q2_output.shape == (_ROWS,)
    assert iq2_output.shape == (_ROWS,)
    assert metrics["q2_k_ms"] >= 0.0
    assert metrics["iq2_xxs_ms"] >= 0.0
