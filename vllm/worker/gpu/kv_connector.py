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
