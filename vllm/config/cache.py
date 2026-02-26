# SPDX-License-Identifier: Apache-2.0
class CacheConfig:
    def __init__(self, block_size: int, gpu_memory_utilization: float, swap_space: int, cache_dtype: str = "auto"):
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.cache_dtype = cache_dtype