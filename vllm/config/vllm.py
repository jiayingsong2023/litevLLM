# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
from .model import ModelConfig
from .cache import CacheConfig
from .scheduler import SchedulerConfig
from .load import LoadConfig

class VllmConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        load_config: LoadConfig,
        quant_config: Optional[Any] = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.load_config = load_config
        self.quant_config = quant_config
