# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, Dict

from vllm.inputs import TextPrompt, TokensPrompt, PromptInput
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.config import ModelConfig

from .params import ChatParams, TokenizeParams

class BaseRenderer(ABC):
    """
    Base class for input rendering and template processing.
    Simplified for LitevLLM architecture.
    """
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._async_tokenizer: Optional[AsyncMicrobatchTokenizer] = None

    @property
    @abstractmethod
    def tokenizer(self) -> Optional[TokenizerLike]:
        raise NotImplementedError

    def get_tokenizer(self) -> TokenizerLike:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return tokenizer

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        if self._async_tokenizer is None:
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())
        return self._async_tokenizer

    def render_completion(self, raw: Union[str, List[int]]) -> Union[TextPrompt, TokensPrompt]:
        """Convert raw input to a structured prompt."""
        if isinstance(raw, str):
            return TextPrompt(prompt=raw)
        elif isinstance(raw, list):
            return TokensPrompt(prompt_token_ids=raw)
        raise TypeError("Input must be string or list of tokens.")

    def render_completions(self, prompt_input: Optional[Union[str, List[Any]]] = None) -> List[Union[TextPrompt, TokensPrompt]]:
        if prompt_input is None:
            raise ValueError("Empty prompt input.")
        
        # Handle list of inputs or single input
        if isinstance(prompt_input, (str, list)) and not isinstance(prompt_input[0], (str, int)):
             inputs = prompt_input
        else:
             inputs = [prompt_input]
             
        return [self.render_completion(p) for p in inputs]

    @abstractmethod
    def render_messages(
        self,
        messages: List[Dict[str, str]],
        params: ChatParams,
    ) -> Tuple[List[Any], Union[TextPrompt, TokensPrompt]]:
        """Render chat messages using a template."""
        raise NotImplementedError

    def tokenize_prompt(
        self,
        prompt: Union[TextPrompt, TokensPrompt],
        params: TokenizeParams,
    ) -> TokensPrompt:
        """Helper to ensure prompt is tokenized."""
        if "prompt_token_ids" not in prompt:
            tokenizer = self.get_tokenizer()
            ids = tokenizer.encode(prompt["prompt"])
            return TokensPrompt(prompt_token_ids=ids, **prompt)
        return prompt

    async def tokenize_prompt_async(
        self,
        prompt: Union[TextPrompt, TokensPrompt],
        params: TokenizeParams,
    ) -> TokensPrompt:
        if "prompt_token_ids" not in prompt:
            tokenizer = self.get_async_tokenizer()
            ids = await tokenizer.encode(prompt["prompt"])
            return TokensPrompt(prompt_token_ids=ids, **prompt)
        return prompt
