# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from vllm.sampling_params import SamplingParams


@dataclass
class RequestState:
    request_id: str
    prompt: str
    guarded_prompt: str
    input_ids: list[int]
    sampling_params: SamplingParams
    lora_id: Optional[str] = None
    lora_int_id: Optional[int] = None
    lora_path: Optional[str] = None
    rng: Optional[torch.Generator] = None
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False
    slot_idx: Optional[int] = None
    seq_len: int = 0
    is_prefill: bool = True
    linear_attn_carry: list[torch.Tensor | None] = field(default_factory=list)
    linear_conv_carry: list[torch.Tensor | None] = field(default_factory=list)
    low_info_hits: int = 0
    is_chinese_capital_question: bool = False
    capital_question_bias_token_ids: list[int] = field(default_factory=list)
    anti_template_token_ids: list[int] = field(default_factory=list)
    structured_output_constraint: Any = None
    service_class: str = "latency"
    multi_modal_data: Any = None
    is_multimodal: bool = False
    is_multimodal_lora: bool = False

    def to_engine_request(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input_ids": self.input_ids,
            "generated_ids": self.generated_ids,
            "sampling_params": self.sampling_params,
            "finished": self.finished,
            "prompt": self.prompt,
            "guarded_prompt": self.guarded_prompt,
            "slot_idx": self.slot_idx,
            "seq_len": self.seq_len,
            "is_prefill": self.is_prefill,
            "lora_id": self.lora_id,
            "lora_int_id": self.lora_int_id,
            "lora_path": self.lora_path,
            "rng": self.rng,
            "linear_attn_carry": self.linear_attn_carry,
            "linear_conv_carry": self.linear_conv_carry,
            "low_info_hits": self.low_info_hits,
            "is_chinese_capital_question": self.is_chinese_capital_question,
            "capital_question_bias_token_ids": self.capital_question_bias_token_ids,
            "anti_template_token_ids": self.anti_template_token_ids,
            "structured_output_constraint": self.structured_output_constraint,
            "service_class": self.service_class,
            "multi_modal_data": self.multi_modal_data,
            "is_multimodal": self.is_multimodal,
            "is_multimodal_lora": self.is_multimodal_lora,
        }
