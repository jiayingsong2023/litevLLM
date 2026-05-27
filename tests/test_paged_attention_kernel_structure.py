# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from vllm.kernels.triton import paged_attention


def _kernel_source() -> str:
    return Path(paged_attention.__file__).read_text(encoding="utf-8")


def test_int4_qk_reduction_keeps_low_high_temporaries_separate() -> None:
    src = _kernel_source()

    assert "q_low[None, :] * k_l + q_high[None, :] * k_h" not in src
    assert "qk_low = tl.sum(q_low[None, :] * k_l, axis=1)" in src
    assert "qk_high = tl.sum(q_high[None, :] * k_h, axis=1)" in src
    assert "qk = (qk_low + qk_high) * scale" in src


def test_int4_v_accumulation_does_not_keep_both_unpack_tiles_named() -> None:
    src = _kernel_source()

    assert "v_l =" not in src
    assert "v_h =" not in src

    low_unpack = src.index("v_low_unpacked =")
    low_accum = src.index(
        "acc_low = acc_low * alpha + tl.sum(p[:, None] * v_low_unpacked, axis=0)"
    )
    high_unpack = src.index("v_high_unpacked =")
    high_accum = src.index(
        "acc_high = acc_high * alpha + tl.sum(p[:, None] * v_high_unpacked, axis=0)"
    )
    assert low_unpack < low_accum < high_unpack < high_accum
