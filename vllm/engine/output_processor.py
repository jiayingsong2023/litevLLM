# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Any, Dict, List, Optional
import torch

def _decode_generated_text(tokenizer: Any, token_ids: List[int], sampling_params: Any) -> str:
    skip = getattr(sampling_params, "skip_special_tokens", True)
    spaces = getattr(sampling_params, "spaces_between_special_tokens", False)
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip,
            spaces_between_special_tokens=spaces,
            clean_up_tokenization_spaces=True,
        )
    except TypeError:
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=skip,
                spaces_between_special_tokens=spaces,
            )
        except TypeError:
            try:
                return tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip,
                    clean_up_tokenization_spaces=True,
                )
            except TypeError:
                return tokenizer.decode(token_ids, skip_special_tokens=skip)

class OutputProcessor:
    def __init__(self, tokenizer: Any, model_path: str):
        self.tokenizer = tokenizer
        self.model_path = model_path

    def apply_prompt_guard(self, prompt: str) -> str:
        return prompt

    def get_anti_template_token_ids(self) -> List[int]:
        return []

    def get_capital_question_bias_token_ids(self, prompt: str) -> List[int]:
        return []

    def is_chinese_capital_question(self, prompt: str) -> bool:
        return False

    def apply_context_bias(self, logits: torch.Tensor, generated_ids: List[int], sampling_params: Any, bias_token_ids: List[int], is_capital_question: bool) -> torch.Tensor:
        return logits

    def should_early_stop(self, generated_ids: List[int], partial_text: str) -> bool:
        return False

    def cleanup_output_text(self, text: str) -> str:
        return text

class DefaultOutputProcessor(OutputProcessor):
    pass

class Qwen35OutputProcessor(OutputProcessor):
    """Placeholder for future Qwen-wide policies.

    Qwen3.5-35B-specific prompt shaping, token biasing, and cleanup rules were
    removed from the lite-only mainline.
    """
    pass

def get_output_processor(model_path: str, tokenizer: Any) -> OutputProcessor:
    b = os.path.basename(os.path.abspath(model_path)).lower()
    if "qwen3.5" in b:
        return Qwen35OutputProcessor(tokenizer, model_path)
    return DefaultOutputProcessor(tokenizer, model_path)
