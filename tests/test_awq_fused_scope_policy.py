# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.tensor import (
    AWQExecutionPolicy,
    should_allow_awq_fused,
    should_use_awq_fused_path,
)


def _dummy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros((1, 128), dtype=torch.bfloat16)
    qweight = torch.zeros((128, 16), dtype=torch.int32)
    scales = torch.ones((128, 4), dtype=torch.float16)
    return x, qweight, scales


def test_should_allow_awq_fused_respects_attention_only_scope() -> None:
    policy = AWQExecutionPolicy(
        prefer_fused=True,
        allow_dense_cache=True,
        cache_scope="all",
        fused_scope="attention_only",
    )
    allow_attn, _ = should_allow_awq_fused("model.layers.0.self_attn.q_proj", policy)
    allow_mlp, reason_mlp = should_allow_awq_fused("model.layers.0.mlp.gate_proj", policy)
    assert allow_attn is True
    assert allow_mlp is False
    assert reason_mlp == "fused_scope_attention_only_non_attn"


def test_should_use_awq_fused_path_short_circuits_on_scope_off() -> None:
    x, qweight, scales = _dummy_inputs()
    policy = AWQExecutionPolicy(
        prefer_fused=True,
        allow_dense_cache=True,
        cache_scope="all",
        fused_scope="off",
    )
    use_fused, reason = should_use_awq_fused_path(
        x=x,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        group_size=32,
        prefix="model.layers.0.self_attn.q_proj",
        policy=policy,
    )
    assert use_fused is False
    assert reason == "fused_scope_off"

