# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, Optional, Type, Union
from vllm.logger import init_logger
from .protocol import TokenizerLike

logger = init_logger(__name__)

class TokenizerRegistry:
    """
    Registry for managing and caching different tokenizer implementations.
    Simplified for LitevLLM to support multi-vendor tokenizers (HF, Mistral, etc.)
    """
    _CACHE: Dict[str, TokenizerLike] = {}

    @staticmethod
    def get_tokenizer_cls(model_config: Any) -> Type[TokenizerLike]:
        model_name = model_config.model.lower()
        
        # 1. Check for specialized implementations
        if "mistral" in model_name:
            try:
                from vllm.tokenizers.mistral import MistralTokenizer
                return MistralTokenizer
            except ImportError:
                logger.warning("Mistral tokenizer dependencies not found, falling back to HF.")
        
        if "grok" in model_name:
            try:
                from vllm.tokenizers.grok2 import Grok2Tokenizer
                return Grok2Tokenizer
            except ImportError:
                pass

        # 2. Default to HuggingFace implementation
        from vllm.tokenizers.hf import CachedHfTokenizer
        return CachedHfTokenizer

    @classmethod
    def get_tokenizer(cls, model_config: Any, **kwargs) -> TokenizerLike:
        model_path = model_config.model
        if model_path not in cls._CACHE:
            tokenizer_cls = cls.get_tokenizer_cls(model_config)
            logger.info(f"Loading tokenizer {model_path} using {tokenizer_cls.__name__}")
            
            cls._CACHE[model_path] = tokenizer_cls.from_pretrained(
                model_path,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True),
                **kwargs
            )
        return cls._CACHE[model_path]

def get_tokenizer(model_config: Any, **kwargs) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(model_config, **kwargs)

def cached_get_tokenizer(model_config: Any, **kwargs) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(model_config, **kwargs)

def cached_tokenizer_from_config(vllm_config: Any) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(vllm_config.model_config)
