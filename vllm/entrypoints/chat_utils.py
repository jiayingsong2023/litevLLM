# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

class ChatTemplateResolutionError(ValueError):
    pass


def make_tool_call_id() -> str:
    return f"call_{random_uuid()}"


ChatTemplateContentFormatOption: TypeAlias = Literal["auto", "text", "openai"]
ChatCompletionMessageParam: TypeAlias = Dict[str, Any]
ConversationMessage: TypeAlias = Dict[str, Any]

def load_chat_template(
    chat_template: Optional[Union[Path, str]] = None,
    *,
    is_literal: bool = False,
) -> Optional[str]:
    """
    Lite implementation of chat template loading.
    """
    if chat_template is None:
        return None

    if is_literal:
        return chat_template

    # If it's a file path
    if os.path.exists(str(chat_template)):
        with open(chat_template, "r") as f:
            return f.read()

    # Fallback to literal if it contains jinja braces
    if "{" in str(chat_template):
        return chat_template

    return None

def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = True,
    **kwargs,
) -> str:
    """
    Apply chat template via HF tokenizer.
    """
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **kwargs
    )
