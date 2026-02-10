# SPDX-License-Identifier: Apache-2.0

import torch

def tensor_force_quant(n_dims: int) -> bool:
    """
    Lite implementation of tensor_force_quant logic.
    Forces quantization if the tensor has more than 1 dimension.
    """
    return n_dims > 1

def quant_noise(x: torch.Tensor, p: float, block_size: int) -> torch.Tensor:
    """
    Lite implementation of quant_noise logic.
    Applies simulated quantization noise to a tensor.
    
    Args:
        x: Input tensor
        p: Dropout probability for noise
        block_size: Block size for noise application
    """
    if p <= 0:
        return x
    
    # Simple simulated noise: randomly zero out elements or add jitter
    # following the pattern of simulated quantization aware training.
    mask = torch.rand_like(x) < p
    noise = (torch.rand_like(x) - 0.5) * (x.abs().mean() * 0.1)
    return x + mask * noise
