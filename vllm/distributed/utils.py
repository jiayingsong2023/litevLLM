# SPDX-License-Identifier: Apache-2.0

"""litevLLM - Simplified distributed utilities."""

import torch
from typing import Sequence

def ensure_divisibility(numerator: int, denominator: int):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )

def divide(numerator: int, denominator: int):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension. Used for weights."""
    if num_partitions == 1:
        return (tensor,)
    
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """For litevLLM, this always returns (0, num_hidden_layers) since pp_size=1."""
    return (0, num_hidden_layers)
