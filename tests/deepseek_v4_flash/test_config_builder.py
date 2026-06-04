from types import SimpleNamespace

import pytest
from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.serving.config_builder import (
    _is_deepseek_v4_flash_gguf,
    _validate_deepseek_v4_flash_context,
    build_vllm_config,
)


def _deepseek_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepSeekV4FlashForCausalLM"],
    )


def test_deepseek_context_validator_accepts_first_release_sizes() -> None:
    hf_config = _deepseek_hf_config()

    assert _validate_deepseek_v4_flash_context(hf_config, 4096) == 4096
    assert _validate_deepseek_v4_flash_context(hf_config, 8192) == 8192


def test_deepseek_context_validator_rejects_larger_sizes() -> None:
    with pytest.raises(ValueError, match="8192"):
        _validate_deepseek_v4_flash_context(_deepseek_hf_config(), 16384)


def test_deepseek_context_validator_ignores_other_models() -> None:
    hf_config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )

    assert _validate_deepseek_v4_flash_context(hf_config, 32768) == 32768


def test_config_builder_detects_deepseek_v4_flash_gguf_metadata(tmp_path) -> None:
    path = tmp_path / "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )

    assert _is_deepseek_v4_flash_gguf(str(path)) is True


def test_build_vllm_config_caps_deepseek_v4_flash_gguf_context(tmp_path) -> None:
    path = tmp_path / "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )

    cfg = build_vllm_config(str(path), max_model_len=8192)

    assert cfg.model_config.hf_config.model_type == "deepseek_v4"
    assert cfg.model_config.max_model_len == 8192

    with pytest.raises(ValueError, match="8192"):
        build_vllm_config(str(path), max_model_len=16384)
