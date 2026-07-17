from __future__ import annotations

import torch

from vllm.kernels.triton.deepseek_v4_flash import (
    DeepSeekV4AttentionKernelInputs,
    DeepSeekV4CacheUpdateInputs,
    DeepSeekV4CompressedAttentionInputs,
    DeepSeekV4MoEKernelInputs,
    DeepSeekV4OutputKernelInputs,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_deepseek_kernel_scaffolding_exports_expected_contracts() -> None:
    attention_inputs = DeepSeekV4AttentionKernelInputs(
        hidden=torch.zeros(16),
        kv_rows=torch.zeros(1, 16),
        token_idx=0,
    )
    cache_inputs = DeepSeekV4CacheUpdateInputs(
        page_table=torch.zeros(1, dtype=torch.int32),
        kv_row=torch.zeros(16),
        cache_storage=torch.zeros(1, 1, 16),
        logical_row=0,
    )
    moe_inputs = DeepSeekV4MoEKernelInputs(
        hidden=torch.zeros(16),
        expert_ids=torch.tensor([0, 1], dtype=torch.int32),
        expert_weights=torch.tensor([0.75, 0.25], dtype=torch.float32),
    )
    output_inputs = DeepSeekV4OutputKernelInputs(
        streams=torch.zeros(4, 32),
        lm_head_values=torch.zeros(8, 32),
        lm_head_scales=torch.zeros(8, 1),
    )
    compressed_inputs = DeepSeekV4CompressedAttentionInputs(
        raw_page_table_name="raw_page_table",
        compressed_page_table_name="compressed_page_table",
        indexer_page_table_name="indexer_page_table",
        selected_rows_name="selected_rows",
    )

    assert attention_inputs.token_idx == 0
    assert cache_inputs.logical_row == 0
    assert moe_inputs.expert_ids.tolist() == [0, 1]
    assert output_inputs.streams.shape == (4, 32)
    assert compressed_inputs.uses_page_tables is True


def test_deepseek_model_keeps_kernel_execution_disabled_until_kernels_exist() -> None:
    model = DeepSeekV4FlashForCausalLM()

    assert model.kernel_execution_available is False
