# SPDX-License-Identifier: Apache-2.0
class EncoderCacheManager:
    def __init__(self, cache_size):
        self.usage = 0.0
    def free(self, *args, **kwargs): pass
    def check_and_update_cache(self, *args, **kwargs): return False
    def can_allocate(self, *args, **kwargs): return True
    def allocate(self, *args, **kwargs): pass
    def get_freed_mm_hashes(self): return []
    def reset(self): pass

class EncoderDecoderCacheManager(EncoderCacheManager): pass

def compute_encoder_budget(*args, **kwargs): return 0, 0
