# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, TypeAlias, TypedDict

import torch

from vllm.inputs import TokensPrompt

ScoreContentPartParam: TypeAlias = dict[str, Any]


class ScoreMultiModalParam(TypedDict, total=False):
    content: list[ScoreContentPartParam]


def _validate_score_input_lens(
    data_1: list[str] | list[ScoreContentPartParam],
    data_2: list[str] | list[ScoreContentPartParam],
) -> None:
    if not data_1 or not data_2:
        raise ValueError("score inputs cannot be empty")
    if len(data_1) != len(data_2) and len(data_1) != 1:
        raise ValueError("data_1 and data_2 must have same length or data_1 length 1")


def compress_token_type_ids(token_type_ids: list[int]) -> list[int]:
    return token_type_ids


def get_score_prompt(
    model_config: Any,
    data_1: str | ScoreContentPartParam,
    data_2: str | ScoreContentPartParam,
    tokenizer: Any,
    tokenization_kwargs: dict[str, Any],
    score_template: str | None = None,
) -> tuple[str, TokensPrompt]:
    del model_config, score_template
    text_1 = str(data_1)
    text_2 = str(data_2)
    full_prompt = f"{text_1}\n{text_2}"
    token_ids = tokenizer.encode(full_prompt, **tokenization_kwargs)
    return full_prompt, TokensPrompt(prompt=full_prompt, prompt_token_ids=token_ids)


def _cosine_similarity(
    tokenizer: Any,
    embed_1: list[Any],
    embed_2: list[Any],
) -> list[Any]:
    del tokenizer
    results: list[Any] = []
    for lhs, rhs in zip(embed_1, embed_2):
        lhs_vec = getattr(getattr(lhs, "outputs", None), "data", [1.0])
        rhs_vec = getattr(getattr(rhs, "outputs", None), "data", [1.0])
        lhs_tensor = torch.as_tensor(lhs_vec, dtype=torch.float32)
        rhs_tensor = torch.as_tensor(rhs_vec, dtype=torch.float32)
        denom = torch.norm(lhs_tensor) * torch.norm(rhs_tensor)
        score = 0.0 if denom.item() == 0 else float(torch.dot(lhs_tensor, rhs_tensor) / denom)

        output = type("ScoreOutput", (), {"score": score})()
        wrapper = type("ScoreWrapper", (), {"outputs": output, "prompt_token_ids": []})()
        results.append(wrapper)
    return results
