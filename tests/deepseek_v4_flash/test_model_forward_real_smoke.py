from pathlib import Path

import pytest
import torch

from vllm.model_executor.model_loader import get_model
from vllm.serving.config_builder import build_vllm_config

MODEL = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not MODEL.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_embedding_output_projection_smoke_returns_finite_logits() -> None:
    model = get_model(build_vllm_config(str(MODEL), max_model_len=4096))
    try:
        logits = model.forward(torch.tensor([1], dtype=torch.long))

        assert model.limited_forward_smoke_only is True
        assert logits.shape == (1, model.shape.vocab_size)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
    finally:
        model.close()
