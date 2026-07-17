# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.layers.quantization.tensor import (
    AWQExecutionPolicy,
    PackedInt4Weight,
    _default_awq_dense_fallback_max_gb,
    awq_decode_gemv_enabled,
    resolve_awq_execution_policy,
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
    allow_mlp, reason_mlp = should_allow_awq_fused(
        "model.layers.0.mlp.gate_proj", policy
    )
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


def test_awq_decode_gemv_helper_prefers_kernel_policy_over_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "0")
    config = SimpleNamespace(kernel_policy={"awq_decode_gemv": True})

    assert awq_decode_gemv_enabled(config) is True


def test_resolve_awq_execution_policy_prefers_kernel_policy_over_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_SCOPE", "off")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM", "0")
    monkeypatch.setenv("FASTINFERENCE_AWQ_CACHE_SCOPE", "off")
    x, _, _ = _dummy_inputs()
    config = SimpleNamespace(
        kernel_policy={
            "awq_fused_scope": "all",
            "awq_fused_gemm": True,
            "awq_cache_scope": "attention_only",
            "awq_dense_fallback_cache": True,
        }
    )

    policy = resolve_awq_execution_policy(
        "model.layers.0.self_attn.q_proj",
        x,
        "gemma4_31b_q4",
        config=config,
    )

    assert policy.prefer_fused is True
    assert policy.allow_dense_cache is True
    assert policy.cache_scope == "attention_only"
    assert policy.fused_scope == "all"


def test_default_awq_dense_fallback_max_gb_ignores_dense_mlp_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_DENSE_MLP", "1")
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        Mock(return_value=SimpleNamespace(total_memory=48 * (1024**3))),
    )

    assert _default_awq_dense_fallback_max_gb() == 14.0


def test_packed_int4_weight_matmul_passes_runtime_kernel_policy(monkeypatch) -> None:
    qweight = torch.zeros((5376, 2688), dtype=torch.uint8)
    scales = torch.ones((5376, 672), dtype=torch.float16)
    weight = PackedInt4Weight(
        qweight,
        scales,
        group_size=32,
        original_shape=(5376, 21504),
        prefix="layers.0.mlp.down_proj",
        profile_hint="gemma4_31b_q4",
    )
    x = torch.zeros((1, 21504), dtype=torch.bfloat16)
    inf_config = SimpleNamespace(
        kernel_policy={
            "awq_decode_gemv": True,
            "gemma4_dense_down_proj": True,
        }
    )
    expected = torch.full((1, 5376), 0.25, dtype=torch.bfloat16)
    seen: dict[str, object] = {}

    def _fake_safe(
        a, qweight, scales, group_size, out=None, bias=None, *, config=None, policy=None
    ):
        seen["config"] = config
        seen["policy"] = policy
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm.packed_int4_symmetric_fused_gemm_safe",
        _fake_safe,
    )

    def _raise_dense(*args, **kwargs):
        raise AssertionError("dense fallback should be bypassed for runtime policy")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.tensor.dequantize_symmetric_packed_int4_pytorch",
        _raise_dense,
    )

    y = weight.matmul(x, config=inf_config)
    assert torch.equal(y, expected)
    assert seen["config"] is inf_config
    assert seen["policy"] is None
