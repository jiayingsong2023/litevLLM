# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vllm.model_executor.models.gemma4 import Gemma4MoeExpertsLite


def _tiny_moe_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4,
        moe_intermediate_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        hidden_act="silu",
        hidden_activation="silu",
        intermediate_size=2,
    )


def test_gemma4_awq_streaming_uses_hidden_dtype_by_default(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(gemma4_moe_expert_cache_size=0),
    )
    monkeypatch.setattr(experts, "_has_awq_packed_expert_major", lambda: True)

    seen_dtypes: list[torch.dtype] = []

    def fake_materialize(
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del expert_id
        seen_dtypes.append(dtype)
        w1 = torch.ones((4, 4), device=device, dtype=dtype)
        w2 = torch.ones((4, 2), device=device, dtype=dtype)
        return w1, w2

    monkeypatch.setattr(experts, "_materialize_one_expert_awq", fake_materialize)

    x = torch.ones((1, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert out.dtype == torch.bfloat16
    assert seen_dtypes == [torch.bfloat16]


def test_gemma4_awq_streaming_can_force_fp32_compute(monkeypatch: Any) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_compute_dtype="fp32",
        ),
    )
    monkeypatch.setattr(experts, "_has_awq_packed_expert_major", lambda: True)

    seen_dtypes: list[torch.dtype] = []

    def fake_materialize(
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del expert_id
        seen_dtypes.append(dtype)
        w1 = torch.ones((4, 4), device=device, dtype=dtype)
        w2 = torch.ones((4, 2), device=device, dtype=dtype)
        return w1, w2

    monkeypatch.setattr(experts, "_materialize_one_expert_awq", fake_materialize)

    x = torch.ones((1, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert out.dtype == torch.bfloat16
    assert seen_dtypes == [torch.float32]
