# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


class DeepSeekV4FlashDirectRuntime:
    backend_name = "deepseek_v4_flash_direct"

    def __init__(
        self,
        *,
        model: Any,
        model_config: Any,
        runtime_config: Any,
        tokenizer: Any | None,
        device: torch.device | str,
        observer: Any | None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.observer = observer
        self.max_model_len = int(getattr(model_config, "max_model_len", 4096))

    def prepare(self) -> None:
        prepare = getattr(self.model, "prepare_for_serving", None)
        if prepare is not None and self.device.type == "cuda":
            prepare(context_length=self.max_model_len, device=self.device)

    def generate(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> RequestOutput:
        if lora_request is not None:
            raise ValueError("DeepSeek V4 Flash direct runtime does not support LoRA")
        if multi_modal_data is not None:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime does not support multimodal input"
            )
        if self.tokenizer is None:
            raise RuntimeError("DeepSeek V4 Flash direct runtime requires a tokenizer")
        self._validate_sampling(sampling_params)
        max_tokens = int(sampling_params.max_tokens or 1)
        prompt_token_ids, input_ids = self._encode_prompt(prompt)
        output_ids = self.model.generate_greedy_kernel(
            input_ids,
            max_tokens=max_tokens,
            use_graph=False,
        )
        return self._build_request_output(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            output_ids=output_ids,
            skip_special_tokens=sampling_params.skip_special_tokens,
        )

    def generate_batched(
        self,
        *,
        request_ids: list[str],
        prompts: list[str],
        sampling_params_list: list[SamplingParams],
    ) -> list[RequestOutput]:
        if self.tokenizer is None:
            raise RuntimeError("DeepSeek V4 Flash direct runtime requires a tokenizer")
        if not prompts:
            return []
        if len(request_ids) != len(prompts) or len(prompts) != len(
            sampling_params_list
        ):
            raise ValueError(
                "request_ids, prompts, and sampling_params_list "
                "must have the same length"
            )
        for sampling_params in sampling_params_list:
            self._validate_sampling(sampling_params)
        max_tokens = int(sampling_params_list[0].max_tokens or 1)
        if any(
            int(sampling_params.max_tokens or 1) != max_tokens
            for sampling_params in sampling_params_list
        ):
            raise ValueError(
                "Batched generation requires all sampling_params.max_tokens to be equal"
            )

        prompt_token_ids_list: list[list[int]] = []
        input_ids_list: list[torch.Tensor] = []
        for prompt in prompts:
            prompt_token_ids, input_ids = self._encode_prompt(prompt)
            prompt_token_ids_list.append(prompt_token_ids)
            input_ids_list.append(input_ids)

        output_ids_list = self.model.generate_greedy_kernel_batched(
            input_ids_list,
            max_tokens=max_tokens,
        )

        outputs: list[RequestOutput] = []
        for request_id, prompt, prompt_token_ids, output_ids, sampling_params in zip(
            request_ids,
            prompts,
            prompt_token_ids_list,
            output_ids_list,
            sampling_params_list,
            strict=True,
        ):
            outputs.append(
                self._build_request_output(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    output_ids=output_ids,
                    skip_special_tokens=sampling_params.skip_special_tokens,
                )
            )
        return outputs

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend_name}

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        del clear_prefix_cache

    def _encode_prompt(self, prompt: str) -> tuple[list[int], torch.Tensor]:
        prompt_token_ids = [int(token) for token in self.tokenizer.encode(prompt)]
        if not prompt_token_ids:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            prompt_token_ids = [0 if eos is None else int(eos)]
        return (
            prompt_token_ids,
            torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device),
        )

    def _decode_generated_token_ids(
        self,
        generated_token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        try:
            return self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        except TypeError:
            return self.tokenizer.decode(generated_token_ids)

    def _build_request_output(
        self,
        *,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        output_ids: torch.Tensor,
        skip_special_tokens: bool,
    ) -> RequestOutput:
        generated_token_ids = [
            int(token)
            for token in output_ids[len(prompt_token_ids) :].detach().cpu().tolist()
        ]
        text = self._decode_generated_token_ids(
            generated_token_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=generated_token_ids,
                    cumulative_logprob=0.0,
                )
            ],
            finished=True,
        )

    @staticmethod
    def _validate_sampling(sampling_params: SamplingParams) -> None:
        if int(getattr(sampling_params, "n", 1) or 1) != 1:
            raise ValueError("DeepSeek V4 Flash direct runtime supports n=1 only")
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens is None or int(max_tokens) <= 0:
            raise ValueError("DeepSeek V4 Flash direct runtime requires max_tokens > 0")
        if float(getattr(sampling_params, "temperature", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        if float(getattr(sampling_params, "top_p", 1.0) or 1.0) != 1.0:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        top_k = int(getattr(sampling_params, "top_k", -1) or -1)
        if top_k not in (-1, 0):
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        if getattr(sampling_params, "structured_outputs", None) is not None:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime does not support structured outputs"
            )
