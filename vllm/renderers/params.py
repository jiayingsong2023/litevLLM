# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption


def merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    base = {} if defaults is None else dict(defaults)
    extra = {} if overrides is None else dict(overrides)
    base.update({k: v for k, v in extra.items() if v not in unset_values})
    return base


@dataclass(frozen=True)
class ChatParams:
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    chat_template_kwargs: dict[str, Any] | None = None

    def with_defaults(
        self, default_chat_template_kwargs: dict[str, Any] | None
    ) -> "ChatParams":
        if not default_chat_template_kwargs:
            return self
        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                default_chat_template_kwargs, self.chat_template_kwargs
            ),
        )

    def get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        return {} if self.chat_template_kwargs is None else dict(self.chat_template_kwargs)


@dataclass(frozen=True)
class TokenizeParams:
    max_total_tokens: int | None
    max_output_tokens: int = 0
    add_special_tokens: bool = True
    needs_detokenization: bool = False
    truncate_prompt_tokens: int | None = None

    def get_encode_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"add_special_tokens": self.add_special_tokens}
        if self.truncate_prompt_tokens is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.truncate_prompt_tokens
        return kwargs
