# SPDX-License-Identifier: Apache-2.0
"""Slice dequant vs full dequant consistency (Q4_0 roundtrip via gguf.quantize)."""
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, dequantize, quant_shape_to_byte_shape, quantize

from vllm.model_executor.moe_gguf_packed import (
    dequant_packed_rows_to_fp16,
    numpy_gguf_data_to_packed_2d,
)


def test_numpy_gguf_data_to_packed_2d_matches_full_dequant_q4_0():
    """Q4_0: full blob dequant equals concat of per-row-block slice dequants."""
    qt = GGMLQuantizationType.Q4_0
    logical = (3, 4, 256)  # E, I, H — last dim multiple of 32
    w = np.random.randn(*logical).astype(np.float32)
    packed = quantize(w, qt)
    assert tuple(packed.shape) == tuple(quant_shape_to_byte_shape(logical, qt))

    full = np.asarray(dequantize(packed, qt))
    full = full.reshape(logical)

    p2d = numpy_gguf_data_to_packed_2d(packed, logical, int(qt))
    assert p2d.shape == (3 * 4, packed.shape[-1])

    torch_p = torch.from_numpy(np.ascontiguousarray(p2d))
    moe_inter, hidden = logical[1], logical[2]
    for e in range(logical[0]):
        r0, r1 = e * moe_inter, (e + 1) * moe_inter
        part = dequant_packed_rows_to_fp16(torch_p, r0, r1, hidden, int(qt)).cpu().numpy()
        np.testing.assert_allclose(part, full[e], rtol=1e-2, atol=0.05)


def test_dequant_packed_rows_matches_full_q4_0():
    qt = GGMLQuantizationType.Q4_0
    n_rows, n_cols = 20, 256
    w = np.random.randn(n_rows, n_cols).astype(np.float32)
    packed = quantize(w, qt)
    full_f = np.asarray(dequantize(packed, qt)).reshape(n_rows, n_cols)

    tp = torch.from_numpy(np.ascontiguousarray(packed))
    for r0, r1 in [(0, 7), (7, 20), (3, 4)]:
        got = dequant_packed_rows_to_fp16(tp, r0, r1, n_cols, int(qt)).cpu().numpy()
        np.testing.assert_allclose(got, full_f[r0:r1], rtol=1e-2, atol=0.05)
