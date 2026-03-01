# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, TypedDict, Union

class _CommonKeys(TypedDict, total=False):
    multi_modal_data: Optional[dict]

class TextPrompt(_CommonKeys):
    prompt: str

class TokensPrompt(_CommonKeys):
    prompt_token_ids: List[int]
    prompt: Optional[str]

PromptInput = Union[str, TextPrompt, TokensPrompt]
