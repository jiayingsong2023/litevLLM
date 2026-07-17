# SPDX-License-Identifier: Apache-2.0
from typing import TypedDict


class _CommonKeys(TypedDict, total=False):
    multi_modal_data: dict | None


class TextPrompt(_CommonKeys):
    prompt: str


class TokensPrompt(_CommonKeys):
    prompt_token_ids: list[int]
    prompt: str | None


class EmbedsPrompt(_CommonKeys):
    prompt_embeds: list[float]
    prompt: str | None


PromptInput = str | TextPrompt | TokensPrompt | EmbedsPrompt
PromptType = PromptInput
