# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable

# Placeholder for missing config function in LitevLLM
def get_cached_compilation_config():
    return None

from vllm.logger import init_logger
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

logger = init_logger(__name__)

class CustomOp(nn.Module):
    def __init__(self, enforce_enable: bool = False):
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward()

    def enabled(self) -> bool:
        config = get_cached_compilation_config()
        if config is None:
            return True
        return hasattr(config, "custom_ops")

    def dispatch_forward(self):
        # In LitevLLM, we prioritize native or hip paths directly
        if torch.version.hip:
            return self.forward_hip
        return self.forward_native

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    @staticmethod
    def register(name: str):
        def wrapper(cls):
            return cls
        return wrapper
