import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.config import DEEPSEEK_V4_FLASH_SHAPE
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 Flash GPU runtime state requires CUDA/ROCm",
)


def test_gpu_request_state_constructs_cuda_request_local_caches() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    config = DeepSeekV4FlashGPUCacheConfig(
        context_length=4096,
        hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
        batch_size=1,
        dtype=dtype,
        device=device,
    )

    state = DeepSeekV4FlashGPURequestState(config)

    assert state.token_position == 0
    assert state.config.hidden_size == DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    assert state.config.kv_width == DEEPSEEK_V4_FLASH_SHAPE.head_dim
    assert state.raw_kv_cache.raw_keys.is_cuda
    assert state.raw_kv_cache.raw_values.is_cuda
    assert state.raw_kv_cache.raw_token_indices.is_cuda
    assert state.raw_kv_cache.raw_keys.device == device
    assert state.raw_kv_cache.raw_values.device == device
    assert state.raw_kv_cache.raw_keys.dtype == dtype
    assert state.raw_kv_cache.raw_values.dtype == dtype
    assert state.raw_kv_cache.raw_keys.shape[-1] == DEEPSEEK_V4_FLASH_SHAPE.head_dim
    assert state.raw_kv_cache.raw_values.shape[-1] == DEEPSEEK_V4_FLASH_SHAPE.head_dim
    assert state.compressed_kv_cache is state.raw_kv_cache
    assert state.compressed_kv_cache.compressed_rows.is_cuda
    assert state.compressed_kv_cache.indexer_rows.is_cuda
    assert state.compressed_kv_cache.compressed_rows.device == device
    assert state.compressed_kv_cache.indexer_rows.device == device
    assert state.compressed_kv_cache.compressed_rows.dtype == dtype
    assert state.compressed_kv_cache.indexer_rows.dtype == dtype
    assert (
        state.compressed_kv_cache.compressed_rows.shape[-1]
        == DEEPSEEK_V4_FLASH_SHAPE.head_dim
    )
    assert state.page_allocator is not None


def test_gpu_request_state_advances_capacity_checks_and_resets() -> None:
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=4096,
            hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
            batch_size=1,
        )
    )

    state.advance_token()

    assert state.token_position == 1
    state.require_capacity(4095)
    with pytest.raises(ValueError, match="context"):
        state.require_capacity(4096)

    state.reset()

    assert state.token_position == 0
    assert torch.all(state.raw_kv_cache.raw_token_indices == -1)
    assert torch.all(state.compressed_kv_cache.compressed_token_indices == -1)


def test_gpu_request_state_rejects_batch_size_other_than_one() -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=4096,
                hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
                batch_size=2,
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_workspace_shape_and_reuse() -> None:
    cfg = DeepSeekV4FlashGPUCacheConfig(
        context_length=1024,
        hidden_size=4096,
        batch_size=1,
        kv_width=512,
        device=torch.device("cuda"),
    )
    state = DeepSeekV4FlashGPURequestState(cfg)
    ws1 = state.moe_workspace(num_experts=6, intermediate_size=2048)
    assert ws1.shape == (6, 2048)
    assert ws1.dtype == torch.float32
    assert ws1.device.type == "cuda"
    ws2 = state.moe_workspace(num_experts=6, intermediate_size=2048)
    assert ws2 is ws1
