# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.logger import init_logger

from .protocol import TokenizerLike

logger = init_logger(__name__)


class TokenizerRegistry:
    """
    Registry for managing and caching different tokenizer implementations.
    Lite runtime tokenizer cache.
    """

    _CACHE: dict[str, TokenizerLike] = {}

    @staticmethod
    def get_tokenizer_cls(model_config: Any) -> type[TokenizerLike]:
        from vllm.tokenizers.hf import CachedHfTokenizer

        return CachedHfTokenizer

    @classmethod
    def get_tokenizer(cls, model_config: Any, **kwargs) -> TokenizerLike:
        model_path = model_config.model
        if model_path not in cls._CACHE:
            tokenizer_cls = cls.get_tokenizer_cls(model_config)
            logger.info(
                "Loading tokenizer %s using %s", model_path, tokenizer_cls.__name__
            )
            cls._CACHE[model_path] = tokenizer_cls.from_pretrained(
                model_path,
                trust_remote_code=getattr(model_config, "trust_remote_code", True),
                **kwargs,
            )
        return cls._CACHE[model_path]


def get_tokenizer(model_config: Any, **kwargs) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(model_config, **kwargs)


def cached_get_tokenizer(model_config: Any, **kwargs) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(model_config, **kwargs)


def cached_tokenizer_from_config(vllm_config: Any) -> TokenizerLike:
    return TokenizerRegistry.get_tokenizer(vllm_config.model_config)


def tokenizer_args_from_config(model_config: Any) -> dict[str, Any]:
    return {
        "trust_remote_code": getattr(model_config, "trust_remote_code", True),
    }
