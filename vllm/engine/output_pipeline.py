# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.engine.output_processor import _decode_generated_text
from vllm.engine.request_state import RequestState
from vllm.outputs import CompletionOutput, RequestOutput


class OutputPipeline:
    def __init__(
        self,
        tokenizer: Any,
        policies: Any,
        sampling_driver: Any,
        *,
        max_model_len: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.policies = policies
        self.sampling_driver = sampling_driver
        self.max_model_len = max_model_len

    def build_abort_output(
        self, request_id: str, request: RequestState
    ) -> RequestOutput:
        completion = CompletionOutput(
            index=0,
            text="",
            token_ids=list(request.generated_ids),
            cumulative_logprob=0.0,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=request.prompt,
            prompt_token_ids=request.input_ids,
            outputs=[completion],
            finished=True,
        )

    def finalize_step(
        self,
        request_id: str,
        request: RequestState,
        next_token: int,
    ) -> RequestOutput:
        sampling_params = request.sampling_params
        eos_ids = self.sampling_driver.completion_eos_ids(request)
        max_tok = int(sampling_params.max_tokens or 16)
        gen_len = len(request.generated_ids)
        min_tok = int(getattr(sampling_params, "min_tokens", 0) or 0)

        reached_context_limit = (
            self.max_model_len is not None and request.seq_len >= self.max_model_len
        )
        finish_reason: str | None = None
        if getattr(sampling_params, "ignore_eos", False):
            if gen_len >= max_tok or reached_context_limit:
                request.finished = True
                finish_reason = "length"
        elif next_token in eos_ids and gen_len >= min_tok:
            request.finished = True
            finish_reason = "stop"
        elif gen_len >= max_tok or reached_context_limit:
            request.finished = True
            finish_reason = "length"

        structured_output_constraint = request.structured_output_constraint
        if structured_output_constraint is not None:
            accepted = structured_output_constraint.on_token(request, next_token)
            if not accepted or structured_output_constraint.should_finish(request):
                request.finished = True

        # Early-stop policies that need partial text are the only reason to
        # decode before the output is actually consumed.
        if not request.finished and self.policies.needs_partial_text_for_early_stop():
            partial_text = _decode_generated_text(
                self.tokenizer, request.generated_ids, sampling_params
            )
            if self.policies.should_early_stop(request.generated_ids, partial_text):
                request.low_info_hits = request.low_info_hits + 1
                if request.low_info_hits >= 2 and gen_len >= max(10, min_tok):
                    request.finished = True
            else:
                request.low_info_hits = 0
        else:
            request.low_info_hits = 0

        # Decode eagerly when finished so the final string is cached; otherwise
        # leave text lazy and only decode when the caller reads it.
        if request.finished:
            display_text = self.policies.cleanup_output_text(
                _decode_generated_text(
                    self.tokenizer, request.generated_ids, sampling_params
                )
            )
            completion = CompletionOutput(
                index=0,
                text=display_text,
                token_ids=request.generated_ids,
                cumulative_logprob=0.0,
            )
        else:
            completion = CompletionOutput(
                index=0,
                text=None,
                token_ids=request.generated_ids,
                cumulative_logprob=0.0,
                tokenizer=self.tokenizer,
                sampling_params=sampling_params,
                text_processor=self.policies.cleanup_output_text,
                finished=False,
            )
        return RequestOutput(
            request_id=request_id,
            prompt=request.prompt,
            prompt_token_ids=request.input_ids,
            outputs=[completion],
            finished=request.finished,
            finish_reason=finish_reason,
        )
