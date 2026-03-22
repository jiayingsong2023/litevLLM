# SPDX-License-Identifier: Apache-2.0
"""LiteConfig must tolerate hf_config.rope_parameters is None (JSON null)."""

from types import SimpleNamespace

from vllm.model_executor.models.lite_config import LiteConfig


def test_rope_parameters_none_uses_empty_dict():
    cfg = SimpleNamespace(
        rope_parameters=None,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=1000,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    lc = LiteConfig(cfg)
    assert lc.rope_parameters == {}
    assert lc.rope_theta == 10000.0


def test_rope_parameters_dict_preserved():
    cfg = SimpleNamespace(
        rope_parameters={"rope_theta": 500000.0, "partial_rotary_factor": 0.5},
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=1000,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
    )
    lc = LiteConfig(cfg)
    assert lc.rope_parameters.get("rope_theta") == 500000.0
    assert abs(lc.partial_rotary_factor - 0.5) < 1e-6
