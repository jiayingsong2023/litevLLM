# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import os
import time
from typing import Any, Optional

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.structured_output_lite import build_structured_output_constraint
from vllm.sampling_params import SamplingParams


class LiteRequestBuilder:
    def __init__(
        self,
        *,
        tokenizer: Any,
        policies: Any,
        device: torch.device,
        num_layers: int,
        max_model_len: int,
        max_tokens_cap: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.policies = policies
        self.device = device
        self.num_layers = num_layers
        self.max_model_len = max_model_len
        self.max_tokens_cap = max_tokens_cap

    def build(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_id: Optional[str] = None,
        lora_int_id: Optional[int] = None,
        lora_path: Optional[str] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        effective_sampling_params = copy.deepcopy(sampling_params)
        max_tokens = effective_sampling_params.max_tokens or 16
        max_tokens = min(max_tokens, self.max_tokens_cap)
        effective_sampling_params.max_tokens = max_tokens
        # Keep env-driven behavior centralized in request building.
        raw = os.environ.get("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "0")
        if int(getattr(effective_sampling_params, "min_tokens", 0) or 0) == 0:
            raw = str(raw).strip().lower()
            if raw not in ("", "0", "false", "off", "no"):
                try:
                    effective_sampling_params.min_tokens = min(max_tokens, max(0, int(raw)))
                except ValueError:
                    effective_sampling_params.min_tokens = min(max_tokens, 1)

        guarded_prompt = self.policies.normalize_prompt(prompt)
        input_ids = self.tokenizer.encode(guarded_prompt)
        if len(input_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt tokens ({len(input_ids)}) exceed/equal max_model_len "
                f"({self.max_model_len}); leave at least one decode token slot."
            )

        rng: Optional[torch.Generator] = None
        seed = getattr(effective_sampling_params, "seed", None)
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(seed))

        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            guarded_prompt=guarded_prompt,
            input_ids=input_ids,
            sampling_params=effective_sampling_params,
            lora_id=lora_id,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
            rng=rng,
            linear_attn_carry=[None] * self.num_layers,
            linear_conv_carry=[None] * self.num_layers,
            is_chinese_capital_question=self.policies.is_chinese_capital_question(prompt),
            capital_question_bias_token_ids=self.policies.capital_question_bias_token_ids(prompt),
            anti_template_token_ids=self.policies.anti_template_token_ids(),
            service_class=str(getattr(effective_sampling_params, "service_class", "latency") or "latency"),
            multi_modal_data=copy.deepcopy(multi_modal_data),
            is_multimodal=bool((multi_modal_data or {}).get("image")),
            is_multimodal_lora=bool(lora_id) and bool((multi_modal_data or {}).get("image")),
        ).to_engine_request()
        request_state["structured_output_constraint"] = build_structured_output_constraint(
            self.tokenizer,
            effective_sampling_params,
        )
        request_state["queued_at"] = time.perf_counter()
        request_state["admitted_at"] = None
        request_state["first_token_at"] = None
        return request_state
