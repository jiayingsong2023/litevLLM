# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.engine.output_processor import _decode_generated_text
from vllm.outputs import CompletionOutput, RequestOutput


class OutputPipeline:
    def __init__(self, tokenizer: Any, policies: Any, sampling_driver: Any):
        self.tokenizer = tokenizer
        self.policies = policies
        self.sampling_driver = sampling_driver

    def build_abort_output(self, request_id: str, request: dict[str, Any]) -> RequestOutput:
        completion = CompletionOutput(
            index=0,
            text="",
            token_ids=list(request["generated_ids"]),
            cumulative_logprob=0.0,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=request["prompt"],
            prompt_token_ids=request["input_ids"],
            outputs=[completion],
            finished=True,
        )

    def finalize_step(
        self,
        request_id: str,
        request: dict[str, Any],
        next_token: int,
    ) -> RequestOutput:
        sampling_params = request["sampling_params"]
        eos_ids = self.sampling_driver.completion_eos_ids(request)
        max_tok = int(sampling_params.max_tokens or 16)
        gen_len = len(request["generated_ids"])
        min_tok = int(getattr(sampling_params, "min_tokens", 0) or 0)

        if getattr(sampling_params, "ignore_eos", False):
            if gen_len >= max_tok:
                request["finished"] = True
        elif next_token in eos_ids and gen_len >= min_tok:
            request["finished"] = True
        elif gen_len >= max_tok:
            request["finished"] = True

        current_text = _decode_generated_text(
            self.tokenizer, request["generated_ids"], sampling_params
        )
        if not request["finished"] and self.policies.should_early_stop(
            request["generated_ids"], current_text
        ):
            request["low_info_hits"] = int(request.get("low_info_hits", 0)) + 1
            if request["low_info_hits"] >= 2 and gen_len >= max(10, min_tok):
                request["finished"] = True
        else:
            request["low_info_hits"] = 0

        display_text = self.policies.cleanup_output_text(current_text)
        completion = CompletionOutput(
            index=0,
            text=display_text,
            token_ids=request["generated_ids"],
            cumulative_logprob=0.0,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=request["prompt"],
            prompt_token_ids=request["input_ids"],
            outputs=[completion],
            finished=request["finished"],
        )
