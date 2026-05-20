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


def test_gemma4_awq_prefill_streaming_batches_selected_expert_materialization(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=False,
            gemma4_moe_prefill_grouped_enabled=False,
            gemma4_moe_batch_materialize_enabled=True,
        ),
    )
    monkeypatch.setattr(experts, "_has_awq_packed_expert_major", lambda: True)

    seen: dict[str, Any] = {}

    def fake_batch_materialize(
        expert_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        seen["expert_ids"] = [int(item) for item in expert_ids.tolist()]
        seen["dtype"] = dtype
        return {
            expert_id: (
                torch.zeros((4, 4), device=device, dtype=dtype),
                torch.ones((4, 2), device=device, dtype=dtype),
            )
            for expert_id in seen["expert_ids"]
        }

    monkeypatch.setattr(
        experts,
        "_materialize_experts_awq_batch",
        fake_batch_materialize,
        raising=False,
    )
    monkeypatch.setattr(
        experts,
        "_materialize_one_expert_awq",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-expert materialization hit")
        ),
    )

    x = torch.ones((3, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((3, 1), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1], [0], [1]], dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert seen == {"expert_ids": [0, 1], "dtype": torch.bfloat16}
    assert torch.equal(out, torch.zeros_like(x))


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


def test_gemma4_awq_streaming_can_select_batched_grouped_streaming_strategy(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_int4_kernel_strategy="batched_grouped_streaming",
        ),
    )
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((2, 4), 31.0, dtype=torch.bfloat16)

    def fake_batched_grouped_streaming(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        del args, kwargs
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_decode_batched_grouped_streaming",
        fake_batched_grouped_streaming,
    )

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)


def test_gemma4_awq_prefill_can_use_packed_int4_grouped_kernel(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_prefill_grouped_enabled=True,
            gemma4_moe_prefill_grouped_min_tokens=2,
        ),
    )
    monkeypatch.setattr(experts, "_has_awq_packed_expert_major", lambda: True)
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((3, 4), 37.0, dtype=torch.bfloat16)
    seen: dict[str, Any] = {}

    def fake_packed_prefill(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        seen["x_shape"] = tuple(args[0].shape)
        seen["topk_weights_shape"] = tuple(args[1].shape)
        seen["topk_ids_shape"] = tuple(args[2].shape)
        seen["gate_up_shape"] = tuple(args[3].shape)
        seen["intermediate_dim"] = kwargs["intermediate_dim"]
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4.gemma4_moe_int4_prefill_grouped",
        fake_packed_prefill,
    )
    monkeypatch.setattr(
        experts,
        "_materialize_expert_weights",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("all-expert materialization hit")
        ),
    )
    monkeypatch.setattr(
        experts,
        "_materialize_one_expert_awq",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-expert materialization hit")
        ),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.gemma4.fused_moe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense fused_moe hit")
        ),
    )

    x = torch.ones((3, 4), dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.8], [0.7], [0.6]],
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[0], [1], [0]], dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)
    assert seen == {
        "x_shape": (3, 4),
        "topk_weights_shape": (3, 1),
        "topk_ids_shape": (3, 1),
        "gate_up_shape": (2, 4, 1),
        "intermediate_dim": 2,
    }


def test_gemma4_awq_prefill_can_use_packed_int4_grouped_fused_kernel(
    monkeypatch: Any,
) -> None:
    experts = Gemma4MoeExpertsLite(
        _tiny_moe_config(),
        quant_config=None,
        prefix="model.layers.0",
        runtime_config=SimpleNamespace(
            gemma4_moe_expert_cache_size=0,
            gemma4_moe_int4_kernel_enabled=True,
            gemma4_moe_prefill_grouped_enabled=True,
            gemma4_moe_prefill_grouped_min_tokens=2,
            gemma4_moe_prefill_grouped_strategy="fused",
        ),
    )
    monkeypatch.setattr(experts, "_has_awq_packed_expert_major", lambda: True)
    experts.gate_up_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.gate_up_proj.scales = torch.empty((2, 4, 1), dtype=torch.float16)
    experts.down_proj.qweight = torch.empty((2, 4, 1), dtype=torch.int32)
    experts.down_proj.scales = torch.empty((2, 2, 1), dtype=torch.float16)

    expected = torch.full((3, 4), 41.0, dtype=torch.bfloat16)

    def fake_fused_prefill(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, bool, str]:
        assert tuple(args[0].shape) == (3, 4)
        assert tuple(args[1].shape) == (3, 1)
        assert tuple(args[2].shape) == (3, 1)
        assert kwargs["intermediate_dim"] == 2
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4."
        "gemma4_moe_int4_prefill_grouped_fused",
        fake_fused_prefill,
    )
    monkeypatch.setattr(
        "vllm.kernels.triton.gemma4_moe_int4.gemma4_moe_int4_prefill_grouped",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("chunked grouped prefill hit")
        ),
    )
    monkeypatch.setattr(
        experts,
        "_materialize_one_expert_awq",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-expert materialization hit")
        ),
    )

    x = torch.ones((3, 4), dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.8], [0.7], [0.6]],
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[0], [1], [0]], dtype=torch.long)

    out = experts(x, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids)

    assert torch.equal(out, expected)
