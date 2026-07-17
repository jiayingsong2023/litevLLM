# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .cache import CacheConfig
from .load import LoadConfig
from .model import ModelConfig
from .scheduler import SchedulerConfig


class VllmConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        load_config: LoadConfig,
        quant_config: Any | None = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.load_config = load_config
        self.quant_config = quant_config
        self.runtime_config = None
        self.model_capabilities = None
        self.runtime_policy_mode = "auto"
