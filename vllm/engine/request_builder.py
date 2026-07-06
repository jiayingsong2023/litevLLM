# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import time
from typing import Any

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
        default_min_new_tokens: int = 0,
        multimodal_processor: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.policies = policies
        self.device = device
        self.num_layers = num_layers
        self.max_model_len = max_model_len
        self.max_tokens_cap = max_tokens_cap
        self.default_min_new_tokens = max(0, int(default_min_new_tokens))
        self.multimodal_processor = multimodal_processor

    def build(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_id: str | None = None,
        lora_int_id: int | None = None,
        lora_path: str | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> RequestState:
        effective_sampling_params = copy.deepcopy(sampling_params)
        max_tokens = effective_sampling_params.max_tokens or 16
        max_tokens = min(max_tokens, self.max_tokens_cap)
        effective_sampling_params.max_tokens = max_tokens
        if int(getattr(effective_sampling_params, "min_tokens", 0) or 0) == 0:
            effective_sampling_params.min_tokens = min(
                max_tokens,
                self.default_min_new_tokens,
            )

        guarded_prompt = self.policies.normalize_prompt(prompt)
        prepared_multi_modal_data = copy.deepcopy(multi_modal_data)
        tokenize_prompt = guarded_prompt
        if prepared_multi_modal_data and self.multimodal_processor is not None:
            prepare_before_tokenize = getattr(
                self.multimodal_processor,
                "prepare_before_tokenize",
                None,
            )
            if prepare_before_tokenize is not None:
                tokenize_prompt, prepared_multi_modal_data = prepare_before_tokenize(
                    guarded_prompt,
                    prepared_multi_modal_data,
                )
        input_ids = self.tokenizer.encode(tokenize_prompt)
        if prepared_multi_modal_data:
            image_token = prepared_multi_modal_data.get("image_token")
            if image_token is not None:
                image_token_text = str(image_token)
                image_token_ids = self.tokenizer.encode(image_token_text)
                if len(image_token_ids) == 1:
                    image_token_id = int(image_token_ids[0])
                elif hasattr(self.tokenizer, "convert_tokens_to_ids"):
                    image_token_id = int(
                        self.tokenizer.convert_tokens_to_ids(image_token_text)
                    )
                else:
                    raise ValueError("image token must encode to exactly one token id")
                prepared_multi_modal_data["image_token_id"] = image_token_id
        if len(input_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt tokens ({len(input_ids)}) exceed/equal max_model_len "
                f"({self.max_model_len}); leave at least one decode token slot."
            )

        rng: torch.Generator | None = None
        seed = getattr(effective_sampling_params, "seed", None)
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(seed))

        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            guarded_prompt=tokenize_prompt,
            input_ids=input_ids,
            sampling_params=effective_sampling_params,
            lora_id=lora_id,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
            rng=rng,
            linear_attn_carry=[None] * self.num_layers,
            linear_conv_carry=[None] * self.num_layers,
            is_chinese_capital_question=self.policies.is_chinese_capital_question(
                prompt
            ),
            capital_question_bias_token_ids=self.policies.capital_question_bias_token_ids(
                prompt
            ),
            anti_template_token_ids=self.policies.anti_template_token_ids(),
            service_class=str(
                getattr(effective_sampling_params, "service_class", "latency")
                or "latency"
            ),
            multi_modal_data=prepared_multi_modal_data,
            is_multimodal=bool((prepared_multi_modal_data or {}).get("image")),
            is_multimodal_lora=bool(lora_id)
            and bool((prepared_multi_modal_data or {}).get("image")),
        )
        request_state.structured_output_constraint = build_structured_output_constraint(
            self.tokenizer,
            effective_sampling_params,
        )
        request_state.queued_at = time.perf_counter()
        request_state.admitted_at = None
        request_state.first_token_at = None
        return request_state
