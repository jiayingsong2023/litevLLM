import json
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.lite_config import LiteConfig


@pytest.fixture
def e2b_text_config():
    return SimpleNamespace(
        **json.load(open("models/gemma-4-E2B-it-AWQ-INT4/config.json"))["text_config"]
    )


@pytest.fixture
def e4b_text_config():
    return SimpleNamespace(
        **json.load(open("models/gemma-4-E4B-it/config.json"))["text_config"]
    )


def test_lite_config_e2b_helpers(e2b_text_config):
    cfg = LiteConfig(e2b_text_config)
    assert cfg.use_double_wide_mlp is True
    assert cfg.num_kv_shared_layers == 20
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.vocab_size_per_layer_input == 262144
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(14) is False
    assert cfg.is_kv_shared_layer(15) is True
    assert cfg.is_kv_shared_layer(34) is True
    assert cfg.effective_intermediate_size(0) == 6144
    assert cfg.effective_intermediate_size(15) == 12288


def test_lite_config_e4b_helpers(e4b_text_config):
    cfg = LiteConfig(e4b_text_config)
    assert cfg.use_double_wide_mlp is False
    assert cfg.num_kv_shared_layers == 18
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(23) is False
    assert cfg.is_kv_shared_layer(24) is True
    assert cfg.effective_intermediate_size(24) == 10240
