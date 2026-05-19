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


def test_gemma4_awq_streaming_uses_int4_kernel_fast_path(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((1, 4), 3.0, dtype=torch.bfloat16)

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4.gemma4_moe_int4_decode",
        fake_decode,
    )
    monkeypatch.setattr(
        experts,
        "_materialize_one_expert_awq",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback hit")),
    )

    x = torch.ones((1, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_single_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="single",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((1, 4), 5.0, dtype=torch.bfloat16)

    def fake_single(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4.gemma4_moe_int4_decode_single_kernel",
        fake_single,
    )

    x = torch.ones((1, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 7.0, dtype=torch.bfloat16)

    def fake_batched(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4.gemma4_moe_int4_decode_batched",
        fake_batched,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_tuned_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_tuned",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 11.0, dtype=torch.bfloat16)

    def fake_batched_tuned(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_tuned",
        fake_batched_tuned,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_chunked_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_chunked",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 13.0, dtype=torch.bfloat16)

    def fake_batched_chunked(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_chunked",
        fake_batched_chunked,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_chunked_pair_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_chunked_pair",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 17.0, dtype=torch.bfloat16)

    def fake_batched_chunked_pair(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_chunked_pair",
        fake_batched_chunked_pair,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_chunked_downpair_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_chunked_downpair",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 19.0, dtype=torch.bfloat16)

    def fake_batched_chunked_downpair(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_chunked_downpair",
        fake_batched_chunked_downpair,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_chunked_splitgate_downpair_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_chunked_splitgate_downpair",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 23.0, dtype=torch.bfloat16)

    def fake_batched_chunked_splitgate_downpair(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_chunked_splitgate_downpair",
        fake_batched_chunked_splitgate_downpair,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_streaming_can_select_batched_grouped_kernel_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_grouped",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 29.0, dtype=torch.bfloat16)

    def fake_batched_grouped(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_grouped",
        fake_batched_grouped,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)
