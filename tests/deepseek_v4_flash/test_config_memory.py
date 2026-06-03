from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
    layer_compress_ratio,
)


def test_flash_shape_matches_target_model() -> None:
    shape = DEEPSEEK_V4_FLASH_SHAPE
    assert shape.num_layers == 43
    assert shape.hidden_size == 4096
    assert shape.vocab_size == 129280
    assert shape.num_attention_heads == 64
    assert shape.num_kv_heads == 1
    assert shape.head_dim == 512
    assert shape.sliding_window == 128
    assert shape.num_experts == 256
    assert shape.num_experts_per_tok == 6


def test_layer_compress_ratio_pattern() -> None:
    assert layer_compress_ratio(0) == 0
    assert layer_compress_ratio(1) == 0
    assert layer_compress_ratio(2) == 4
    assert layer_compress_ratio(3) == 128
    assert layer_compress_ratio(4) == 4
    assert layer_compress_ratio(42) == 4


def test_memory_policy_caps_first_release_context() -> None:
    policy = DeepSeekV4FlashMemoryPolicy()
    assert policy.validate_context_length(4096) == 4096
    assert policy.validate_context_length(8192) == 8192
    try:
        policy.validate_context_length(16384)
    except ValueError as exc:
        assert "8192" in str(exc)
    else:
        raise AssertionError("context above first-release cap must fail")


def test_memory_estimate_increases_with_context() -> None:
    policy = DeepSeekV4FlashMemoryPolicy()
    estimate_4k = policy.estimate_context_bytes(4096)
    estimate_8k = policy.estimate_context_bytes(8192)
    assert estimate_4k.raw_kv_bytes > 0
    assert estimate_4k.compressed_kv_bytes > 0
    assert estimate_8k.total_bytes > estimate_4k.total_bytes
