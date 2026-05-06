# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoTokenizer


def test_local_tinyllama_config_and_tokenizer_roundtrip() -> None:
    model_path = Path(
        os.environ.get("MODEL_TINYLLAMA", "models/TinyLlama-1.1B-Chat-v1.0")
    )
    if not model_path.is_dir():
        pytest.skip(f"local smoke model is not available: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    token_ids = tokenizer.encode("FastInference smoke")
    assert token_ids
    assert isinstance(tokenizer.decode(token_ids), str)
    assert getattr(config, "model_type", None)
