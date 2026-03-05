# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from transformers import AutoConfig, AutoTokenizer


def _pick_real_model_path() -> str | None:
    candidates = [
        os.environ.get("FASTINFERENCE_REAL_MODEL"),
        "models/TinyLlama-1.1B-Chat-v1.0",
        "models/Qwen3.5-9B-GGUF",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


@pytest.mark.slow_test
def test_real_model_lightweight_load_and_tokenizer_roundtrip():
    model_path = _pick_real_model_path()
    if model_path is None:
        pytest.skip(
            "No real model found. Set FASTINFERENCE_REAL_MODEL or place model "
            "under models/."
        )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    assert config is not None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode("FastInference smoke test", add_special_tokens=True)
    assert len(token_ids) > 0

    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
