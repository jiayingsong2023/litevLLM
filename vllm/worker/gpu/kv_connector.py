# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Any
from vllm.config import VllmConfig
from vllm.v1_outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

class KVConnector:
    """Simplified No-op KVConnector for litevLLM."""
    def pre_forward(self, scheduler_output: Any) -> None:
        pass

    def post_forward(
        self, scheduler_output: Any, wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: Any) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass

NO_OP_KV_CONNECTOR = KVConnector()

def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    return NO_OP_KV_CONNECTOR
