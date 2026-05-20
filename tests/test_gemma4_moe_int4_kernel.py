# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.kernels.triton.gemma4_moe_int4 import (
    gemma4_moe_int4_decode,
    gemma4_moe_int4_decode_batched,
    gemma4_moe_int4_decode_batched_chunked,
    gemma4_moe_int4_decode_batched_chunked_downpair,
    gemma4_moe_int4_decode_batched_chunked_pair,
    gemma4_moe_int4_decode_batched_chunked_splitgate_downpair,
    gemma4_moe_int4_decode_batched_grouped,
    gemma4_moe_int4_decode_batched_grouped_streaming,
    gemma4_moe_int4_decode_batched_tuned,
    gemma4_moe_int4_decode_single_kernel,
    gemma4_moe_int4_prefill_grouped,
    gemma4_moe_int4_prefill_grouped_fused,
)
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def _pack_int4_signed(values: torch.Tensor) -> torch.Tensor:
    unsigned = (values.to(torch.int32) + 8).clamp(0, 15)
    rows, cols = unsigned.shape
    assert cols % 8 == 0
    packed = torch.zeros((rows, cols // 8), device=values.device, dtype=torch.int32)
    for nibble in range(8):
        packed |= unsigned[:, nibble::8] << (4 * nibble)
    return packed


def _make_expert_major_weight(
    experts: int,
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert cols % 32 == 0
    q = torch.randint(-8, 8, (experts, rows, cols), device=device, dtype=torch.int32)
    scales = (
        torch.rand((experts, rows, cols // 32), device=device, dtype=torch.float16)
        * 0.15
        + 0.01
    )
    packed = torch.stack([_pack_int4_signed(q[e]) for e in range(experts)], dim=0)
    return packed.contiguous(), scales.contiguous()


def _reference_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    intermediate_dim: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    for token_idx in range(int(x.shape[0])):
        token = x[token_idx : token_idx + 1].float()
        for pos in range(int(topk_ids.shape[1])):
            expert_id = int(topk_ids[token_idx, pos].item())
            w1 = dequantize_symmetric_packed_int4_pytorch(
                gate_up_qweight[expert_id],
                gate_up_scales[expert_id],
                group_size=32,
            ).to(device=x.device, dtype=torch.float32)
            w2 = dequantize_symmetric_packed_int4_pytorch(
                down_qweight[expert_id],
                down_scales[expert_id],
                group_size=32,
            ).to(device=x.device, dtype=torch.float32)
            gu = F.linear(token, w1)
            gate, up = torch.chunk(gu[:, : 2 * intermediate_dim], 2, dim=-1)
            if activation in ("gelu", "gelu_pytorch_tanh"):
                hidden = F.gelu(gate, approximate="tanh") * up
            else:
                hidden = F.silu(gate) * up
            route_weight = topk_weights[token_idx : token_idx + 1, pos].float()
            out[token_idx : token_idx + 1] += F.linear(hidden, w2) * route_weight
    return out.to(x.dtype)


def test_gemma4_moe_int4_decode_matches_reference() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(1234)

    experts = 4
    hidden_dim = 64
    intermediate_dim = 32
    x = torch.randn((1, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[2, 0, 3]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.55, 0.30, 0.15]], device=device, dtype=torch.float16
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    fused, used, reason = gemma4_moe_int4_decode(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
    )
    assert used, reason
    torch.testing.assert_close(fused.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_single_kernel_matches_reference() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(4321)

    experts = 4
    hidden_dim = 64
    intermediate_dim = 32
    x = torch.randn((1, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3]], device=device, dtype=torch.long)
    topk_weights = torch.tensor([[0.65, 0.35]], device=device, dtype=torch.float16)
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    fused, used, reason = gemma4_moe_int4_decode_single_kernel(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
    )
    assert used, reason
    torch.testing.assert_close(fused.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_matches_reference() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(9876)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 32
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    fused, used, reason = gemma4_moe_int4_decode_batched(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
    )
    assert used, reason
    torch.testing.assert_close(fused.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_tuned_matches_batched() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(1357)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 32
    x = torch.randn((4, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor(
        [[1, 3], [3, 0], [2, 1], [4, 3]], device=device, dtype=torch.long
    )
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45], [0.20, 0.80]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )
    batched, used_batched, batched_reason = gemma4_moe_int4_decode_batched(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_tuned, tuned_reason
    assert used_batched, batched_reason
    torch.testing.assert_close(tuned.float(), batched.float(), rtol=1e-5, atol=1e-5)


def test_gemma4_moe_int4_decode_batched_tuned_accepts_tile_overrides() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(2468)

    experts = 4
    hidden_dim = 64
    intermediate_dim = 32
    x = torch.randn((2, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.70, 0.30], [0.45, 0.55]], device=device, dtype=torch.float16
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
        block_h_override=32,
        block_packs_override=16,
    )
    batched, used_batched, batched_reason = gemma4_moe_int4_decode_batched(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_tuned, tuned_reason
    assert used_batched, batched_reason
    torch.testing.assert_close(tuned.float(), batched.float(), rtol=1e-5, atol=1e-5)


def test_gemma4_moe_int4_decode_batched_tuned_rejects_invalid_tile() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    x = torch.randn((2, 64), device=device, dtype=torch.float16)
    topk_ids = torch.zeros((2, 1), device=device, dtype=torch.long)
    topk_weights = torch.ones((2, 1), device=device, dtype=torch.float16)
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        2,
        64,
        64,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        2,
        64,
        32,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=32,
        block_h_override=48,
        block_packs_override=8,
    )

    assert not used
    assert reason == "invalid_tuned_tile"
    assert out is x


def test_gemma4_moe_int4_decode_batched_chunked_matches_tuned() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(8642)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 64
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    chunked, used_chunked, chunked_reason = gemma4_moe_int4_decode_batched_chunked(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
        block_i_override=64,
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_chunked, chunked_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(chunked.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_chunked_supports_gelu_tanh() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(1357)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 64
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    chunked, used, reason = gemma4_moe_int4_decode_batched_chunked(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="gelu_pytorch_tanh",
        block_i_override=64,
    )
    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
        activation="gelu_pytorch_tanh",
    )

    assert used, reason
    torch.testing.assert_close(chunked.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_chunked_pair_matches_tuned() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(9753)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 96
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    paired, used_paired, paired_reason = gemma4_moe_int4_decode_batched_chunked_pair(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
        block_i_override=64,
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_paired, paired_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(paired.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_chunked_downpair_matches_tuned() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(6420)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 96
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    downpair, used_downpair, downpair_reason = (
        gemma4_moe_int4_decode_batched_chunked_downpair(
            x,
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
            intermediate_dim=intermediate_dim,
            activation="silu",
            block_i_override=64,
        )
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_downpair, downpair_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(downpair.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_chunked_splitgate_downpair_matches_tuned() -> (
    None
):
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(7531)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 96
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    splitgate, used_splitgate, splitgate_reason = (
        gemma4_moe_int4_decode_batched_chunked_splitgate_downpair(
            x,
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
            intermediate_dim=intermediate_dim,
            activation="silu",
            block_i_override=64,
        )
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_splitgate, splitgate_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(splitgate.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_grouped_matches_tuned() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(8642)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 96
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    grouped, used_grouped, grouped_reason = gemma4_moe_int4_decode_batched_grouped(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
        block_i_override=64,
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_grouped, grouped_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(grouped.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_batched_grouped_streaming_matches_tuned() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(9753)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 96
    x = torch.randn((3, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor([[1, 3], [3, 0], [2, 1]], device=device, dtype=torch.long)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.40, 0.60], [0.55, 0.45]],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    streaming, used_streaming, streaming_reason = (
        gemma4_moe_int4_decode_batched_grouped_streaming(
            x,
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
            intermediate_dim=intermediate_dim,
            activation="silu",
            block_i_override=64,
        )
    )
    tuned, used_tuned, tuned_reason = gemma4_moe_int4_decode_batched_tuned(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )

    assert used_streaming, streaming_reason
    assert used_tuned, tuned_reason
    torch.testing.assert_close(streaming.float(), tuned.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_decode_rejects_prefill_shape() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    x = torch.randn((2, 64), device=device, dtype=torch.float16)
    topk_ids = torch.zeros((2, 1), device=device, dtype=torch.long)
    topk_weights = torch.ones((2, 1), device=device, dtype=torch.float16)
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        2,
        64,
        64,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        2,
        64,
        32,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_decode(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=32,
        activation="silu",
    )

    assert out is x
    assert used is False
    assert reason == "input_not_m1_2d"


def test_gemma4_moe_int4_decode_batched_rejects_large_prefill_shape() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    x = torch.randn((17, 64), device=device, dtype=torch.float16)
    topk_ids = torch.zeros((17, 1), device=device, dtype=torch.long)
    topk_weights = torch.ones((17, 1), device=device, dtype=torch.float16)
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        2,
        64,
        64,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        2,
        64,
        32,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_decode_batched(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=32,
        activation="silu",
    )

    assert out is x
    assert used is False
    assert reason == "batch_too_large_for_decode"


def test_gemma4_moe_int4_prefill_grouped_accepts_large_prefill_shape() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(2468)

    experts = 4
    hidden_dim = 64
    intermediate_dim = 64
    n_tokens = 17
    x = torch.randn((n_tokens, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.arange(n_tokens, device=device, dtype=torch.long).view(-1, 1) % 4
    topk_weights = torch.ones((n_tokens, 1), device=device, dtype=torch.float16)
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_prefill_grouped(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="silu",
        block_i_override=64,
    )
    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
    )

    assert used, reason
    torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_gemma4_moe_int4_prefill_grouped_fused_matches_reference() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(9753)

    experts = 5
    hidden_dim = 64
    intermediate_dim = 32
    n_tokens = 9
    x = torch.randn((n_tokens, hidden_dim), device=device, dtype=torch.float16)
    topk_ids = torch.tensor(
        [
            [3, 1],
            [1, 4],
            [3, 0],
            [4, 1],
            [0, 3],
            [1, 2],
            [3, 4],
            [2, 0],
            [4, 3],
        ],
        device=device,
        dtype=torch.long,
    )
    topk_weights = torch.tensor(
        [
            [0.65, 0.35],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.25, 0.75],
            [0.40, 0.60],
            [0.70, 0.30],
            [0.52, 0.48],
            [0.33, 0.67],
            [0.58, 0.42],
        ],
        device=device,
        dtype=torch.float16,
    )
    gate_up_qweight, gate_up_scales = _make_expert_major_weight(
        experts,
        2 * intermediate_dim,
        hidden_dim,
        device=device,
    )
    down_qweight, down_scales = _make_expert_major_weight(
        experts,
        hidden_dim,
        intermediate_dim,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_prefill_grouped_fused(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation="gelu_pytorch_tanh",
        block_i_override=32,
        block_h_override=32,
    )
    ref = _reference_moe(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim,
        activation="gelu_pytorch_tanh",
    )

    assert used, reason
    torch.testing.assert_close(out.float(), ref.float(), rtol=4e-2, atol=4e-2)
