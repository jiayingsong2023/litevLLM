# SPDX-License-Identifier: Apache-2.0
class ModelMetrics:
    def __init__(self, *args, **kwargs): pass
    def is_enabled(self): return False
    def get_step_perf_stats_per_gpu(self, *args, **kwargs): return None

class PerfStats: pass
