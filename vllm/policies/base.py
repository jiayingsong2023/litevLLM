# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, List

import torch


@dataclass
class GenerationPolicies:
    backend: Any

    def normalize_prompt(self, prompt: str) -> str:
        return self.backend.apply_prompt_guard(prompt)

    def anti_template_token_ids(self) -> List[int]:
        return self.backend.get_anti_template_token_ids()

    def capital_question_bias_token_ids(self, prompt: str) -> List[int]:
        return self.backend.get_capital_question_bias_token_ids(prompt)

    def is_chinese_capital_question(self, prompt: str) -> bool:
        return self.backend.is_chinese_capital_question(prompt)

    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        sampling_params: Any,
        bias_token_ids: List[int],
        is_capital_question: bool,
    ) -> torch.Tensor:
        return self.backend.apply_context_bias(
            logits,
            generated_ids,
            sampling_params,
            bias_token_ids,
            is_capital_question,
        )

    def should_early_stop(self, generated_ids: List[int], partial_text: str) -> bool:
        return self.backend.should_early_stop(generated_ids, partial_text)

    def cleanup_output_text(self, text: str) -> str:
        return self.backend.cleanup_output_text(text)
