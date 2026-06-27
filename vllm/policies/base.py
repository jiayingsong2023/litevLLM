# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GenerationPolicies:
    backend: Any

    def normalize_prompt(self, prompt: str) -> str:
        return self.backend.apply_prompt_guard(prompt)

    def anti_template_token_ids(self) -> list[int]:
        return self.backend.get_anti_template_token_ids()

    def capital_question_bias_token_ids(self, prompt: str) -> list[int]:
        return self.backend.get_capital_question_bias_token_ids(prompt)

    def is_chinese_capital_question(self, prompt: str) -> bool:
        return self.backend.is_chinese_capital_question(prompt)

    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        sampling_params: Any,
        bias_token_ids: list[int],
        is_capital_question: bool,
    ) -> torch.Tensor:
        return self.backend.apply_context_bias(
            logits,
            generated_ids,
            sampling_params,
            bias_token_ids,
            is_capital_question,
        )

    def should_early_stop(self, generated_ids: list[int], partial_text: str) -> bool:
        return self.backend.should_early_stop(generated_ids, partial_text)

    def needs_partial_text_for_early_stop(self) -> bool:
        return self.backend.needs_partial_text_for_early_stop()

    def cleanup_output_text(self, text: str) -> str:
        return self.backend.cleanup_output_text(text)
