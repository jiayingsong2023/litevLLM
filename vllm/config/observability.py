# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

@dataclass
class ObservabilityConfig:
    otlp_traces_endpoint: str = ""
    collect_model_forward_time: bool = False
    collect_model_execute_time: bool = False
    kv_cache_metrics: bool = False
    kv_cache_metrics_sample: float = 0.0
    enable_mfu_metrics: bool = False
