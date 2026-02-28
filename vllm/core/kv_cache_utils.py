# SPDX-License-Identifier: Apache-2.0
from vllm.kv_cache_interface import KVCacheConfig

def generate_scheduler_kv_cache_config(vllm_config):
    return KVCacheConfig(num_blocks=vllm_config.cache_config.num_gpu_blocks, kv_cache_groups=[])

def get_kv_cache_configs(*args, **kwargs): return []
