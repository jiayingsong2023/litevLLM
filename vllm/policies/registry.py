# SPDX-License-Identifier: Apache-2.0
from typing import Any

from vllm.engine.output_processor import get_output_processor

from .base import GenerationPolicies


def build_generation_policies(model_path: str, tokenizer: Any, _adapter: Any) -> GenerationPolicies:
    backend = get_output_processor(model_path, tokenizer)
    return GenerationPolicies(backend=backend)
