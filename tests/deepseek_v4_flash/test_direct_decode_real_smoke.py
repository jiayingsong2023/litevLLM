from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.model_executor.model_loader import get_model
from vllm.serving.config_builder import build_vllm_config

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_direct_full_model_one_token_decode_returns_logits() -> None:
    model = get_model(build_vllm_config(str(TARGET_GGUF), max_model_len=4096))
    try:
        logits = model.forward_full_reference(torch.tensor([1], dtype=torch.long))

        assert logits.shape == (1, model.shape.vocab_size)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert model.limited_forward_smoke_only is False
    finally:
        model.close()


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_direct_greedy_reference_appends_one_token() -> None:
    model = get_model(build_vllm_config(str(TARGET_GGUF), max_model_len=4096))
    try:
        tokens = model.generate_greedy_reference(
            torch.tensor([1], dtype=torch.long),
            max_tokens=1,
        )

        assert tokens.shape == (2,)
        assert tokens.dtype == torch.long
        assert int(tokens[0]) == 1
        assert 0 <= int(tokens[1]) < model.shape.vocab_size
    finally:
        model.close()
