from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash import DeepSeekV4FlashForCausalLM
from vllm.model_executor.models.registry import ModelRegistry


def test_model_registry_resolves_deepseek_v4_flash() -> None:
    cfg = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4"))

    cls, arch = ModelRegistry.resolve_model_cls(["DeepSeekV4FlashForCausalLM"], cfg)

    assert cls is DeepSeekV4FlashForCausalLM
    assert arch == "DeepSeekV4FlashForCausalLM"


def test_model_registry_infers_deepseek_v4_flash_from_config() -> None:
    cfg = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek4"))

    cls, arch = ModelRegistry.resolve_model_cls([], cfg)

    assert cls is DeepSeekV4FlashForCausalLM
    assert arch == "DeepSeekV4FlashForCausalLM"


def test_deepseek_v4_flash_skeleton_requires_weight_store() -> None:
    model = DeepSeekV4FlashForCausalLM()

    with pytest.raises(RuntimeError, match="attached GGUF weight store"):
        model(torch.tensor([1], dtype=torch.long))


def test_deepseek_v4_flash_limited_smoke_returns_empty_logits() -> None:
    model = DeepSeekV4FlashForCausalLM(weight_store=object())

    logits = model(torch.empty((0,), dtype=torch.long))

    assert logits.shape == (0, model.shape.vocab_size)
    assert logits.dtype == torch.float32


def test_deepseek_v4_flash_limited_smoke_rejects_two_tokens() -> None:
    model = DeepSeekV4FlashForCausalLM(weight_store=object())

    with pytest.raises(
        ValueError,
        match="limited smoke supports one token until full transformer forward",
    ):
        model(torch.tensor([1, 2], dtype=torch.long))
