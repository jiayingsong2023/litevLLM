# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:

    architectures: list[str] | None

    model_type: str

    hidden_size: int

    total_num_attention_heads: int

    vocab_size: int

    num_experts: int

    is_deepseek_mla: bool
