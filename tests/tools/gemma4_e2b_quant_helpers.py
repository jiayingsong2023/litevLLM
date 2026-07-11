"""Shared helpers for E2B asymmetric packed-int4 tests and microbench."""

import torch


def pack_asymmetric_int4(
    weights: torch.Tensor, zeros: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack signed int4 weights and zeros into uint32 tensors."""
    n, k = weights.shape
    assert k % 8 == 0 and n % 8 == 0
    n_groups = k // 32
    assert zeros.shape == (n, n_groups)

    qweight = torch.zeros(n, k // 8, dtype=torch.int32, device=weights.device)
    for i in range(8):
        qweight |= (weights[:, i::8].to(torch.int32) + 8) << (i * 4)

    qzeros = torch.zeros(n // 8, n_groups, dtype=torch.int32, device=zeros.device)
    for i in range(8):
        qzeros |= (zeros[i::8, :].to(torch.int32) + 8) << (i * 4)

    return qweight, qzeros


def make_asymmetric_packed_int4(
    n: int, k: int, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert k % group_size == 0 and n % 8 == 0
    n_groups = k // group_size
    weights = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    zeros = torch.randint(-8, 8, (n, n_groups), dtype=torch.int8, device="cuda")
    scales = torch.randn(n, n_groups, dtype=torch.float16, device="cuda") * 0.01
    qweight, qzeros = pack_asymmetric_int4(weights, zeros)
    return qweight, scales, qzeros
