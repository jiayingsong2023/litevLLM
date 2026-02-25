# SPDX-License-Identifier: Apache-2.0
"""LitevLLM Distributed Utils Stub."""

def divide(numerator, denominator):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator

def get_pp_indices(num_layers, pp_rank, pp_size):
    return 0, num_layers

class StatelessProcessGroup:
    """Mock process group for single card."""
    def __init__(self, *args, **kwargs):
        self.rank = 0
        self.world_size = 1