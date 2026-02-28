# SPDX-License-Identifier: Apache-2.0
class KVCacheBlocks:
    def get_block_ids(self, allow_none=False): return []

class KVCacheManager:
    def __init__(self, **kwargs):
        self.usage = 0.0
        self.empty_kv_cache_blocks = KVCacheBlocks()
    def allocate_slots(self, *args, **kwargs): return KVCacheBlocks()
    def get_computed_blocks(self, *args, **kwargs): return KVCacheBlocks(), 0
    def get_blocks(self, *args, **kwargs): return KVCacheBlocks()
    def get_num_common_prefix_blocks(self, *args, **kwargs): return [0]
    def free(self, *args, **kwargs): pass
    def take_events(self): return []
    def reset_prefix_cache(self): return True
    def make_prefix_cache_stats(self): return None
