# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.model_executor.layers.quantization.tensor import (
    get_awq_runtime_audit_summary,
    get_awq_runtime_stats,
    record_awq_audit_event,
    reset_awq_runtime_stats,
)


def test_awq_audit_summary_keeps_first_fallback_detail_once() -> None:
    reset_awq_runtime_stats()

    for _ in range(3):
        record_awq_audit_event(
            "model.layers.0.mlp.down_proj",
            "dense_fallback",
            shape={"m": 1, "n": 5376, "k": 21504, "group_size": 32},
            dtypes={"x": "bfloat16", "qweight": "uint8", "scales": "float16"},
            decision_path=[
                "gemma4_dense_down_proj=1",
                "awq_decode_gemv=1",
                "safe_wrapper_failed",
            ],
            reason="fused_runtime_fallback",
        )

    summary = get_awq_runtime_audit_summary(limit=8)

    assert summary["events"]["dense_fallback"] == 3
    assert summary["prefixes"]["model.layers.0.mlp.down_proj"]["dense_fallback"] == 3
    assert summary["first_fallbacks"] == [
        {
            "prefix": "model.layers.0.mlp.down_proj",
            "event": "dense_fallback",
            "shape": {"m": 1, "n": 5376, "k": 21504, "group_size": 32},
            "dtypes": {"x": "bfloat16", "qweight": "uint8", "scales": "float16"},
            "decision_path": [
                "gemma4_dense_down_proj=1",
                "awq_decode_gemv=1",
                "safe_wrapper_failed",
            ],
            "reason": "fused_runtime_fallback",
            "count": 3,
        }
    ]

def test_awq_audit_summary_tracks_qkv_projection_path() -> None:
    reset_awq_runtime_stats()

    record_awq_audit_event(
        "model.layers.1.self_attn",
        "qkv_separate_decode",
        shape={"m": 1, "q": 8192, "k": 4096, "v": 4096},
        reason="gemma4_attention_forward_uses_separate_litelinears",
    )
    record_awq_audit_event(
        "model.layers.2.self_attn",
        "qk_fused_decode",
        shape={"m": 1, "q": 16384, "k": 2048},
        reason="attention_k_eq_v",
    )

    summary = get_awq_runtime_audit_summary(limit=8)

    assert summary["qkv_projection_paths"] == {
        "qk_fused_decode": 1,
        "qkv_separate_decode": 1,
    }
    assert summary["prefixes"]["model.layers.1.self_attn"]["qkv_separate_decode"] == 1
    assert summary["prefixes"]["model.layers.2.self_attn"]["qk_fused_decode"] == 1

def test_awq_audit_summary_tracks_mlp_streaming_events() -> None:
    reset_awq_runtime_stats()

    record_awq_audit_event(
        "model.layers.0.mlp",
        "mlp_streaming_fallback",
        shape={"m": 1, "hidden": 5376, "intermediate": 21504},
        reason="disabled",
    )
    record_awq_audit_event(
        "model.layers.0.mlp",
        "mlp_streaming_attempt",
        shape={"m": 1, "hidden": 5376, "intermediate": 21504},
        reason="policy_enabled",
    )

    summary = get_awq_runtime_audit_summary(limit=8)

    assert summary["events"]["mlp_streaming_fallback"] == 1
    assert summary["events"]["mlp_streaming_attempt"] == 1
    assert summary["prefixes"]["model.layers.0.mlp"]["mlp_streaming_fallback"] == 1

def test_gemma4_prefill_down_proj_prefers_fused_path(monkeypatch) -> None:
    import torch

    from vllm.model_executor.layers.quantization.tensor import PackedInt4Weight

    reset_awq_runtime_stats()
    qweight = torch.zeros((5376, 2688), dtype=torch.int32)
    scales = torch.ones((5376, 672), dtype=torch.bfloat16)
    weight = PackedInt4Weight(
        qweight,
        scales,
        group_size=32,
        original_shape=(5376, 21504),
        prefix="layers.0.mlp.down_proj",
        profile_hint="gemma4_31b_q4",
    )
    x = torch.zeros((141, 21504), dtype=torch.bfloat16)
    expected = torch.full((141, 5376), 0.5, dtype=torch.bfloat16)

    def fake_safe(a, qweight, scales, group_size, out=None, bias=None, config=None):
        assert a.shape == (141, 21504)
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm.packed_int4_symmetric_fused_gemm_safe",
        fake_safe,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.tensor.dequantize_symmetric_packed_int4_pytorch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefill down_proj should not dense materialize")
        ),
    )

    config = {
        "kernel_policy": {
            "gemma4_dense_down_proj": True,
            "awq_decode_gemv": True,
        }
    }
    y = weight.matmul(x, config=config)

    assert torch.equal(y, expected)
    summary = get_awq_runtime_audit_summary()
    assert summary["events"].get("dense_fallback", 0) == 0

def test_mlp_streaming_helper_default_disabled_records_fallback() -> None:
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.models._fused_awq_pair import (
        try_fused_awq_mlp_streaming,
    )

    reset_awq_runtime_stats()
    x = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
    proj = SimpleNamespace(
        qweight=torch.zeros((64, 4), dtype=torch.int32),
        scales=torch.ones((64, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
        input_size=32,
        output_size=64,
        bias=None,
        prefix="model.layers.0.mlp.gate_proj",
    )
    down = SimpleNamespace(
        qweight=torch.zeros((32, 8), dtype=torch.int32),
        scales=torch.ones((32, 2), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
        input_size=64,
        output_size=32,
        bias=None,
        prefix="model.layers.0.mlp.down_proj",
    )

    out = try_fused_awq_mlp_streaming(
        x,
        proj,
        proj,
        down,
        activation="silu",
        lora_mapping=None,
        inf_config={"kernel_policy": {"awq_mlp_streaming_fusion": False}},
        prefix="model.layers.0.mlp",
    )

    assert out is None
    summary = get_awq_runtime_audit_summary(limit=8)
    assert summary["events"]["mlp_streaming_fallback"] == 1
    assert summary["first_fallbacks"][0]["reason"] == "disabled"

def test_mlp_streaming_helper_policy_enabled_reports_required_design() -> None:
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.models._fused_awq_pair import (
        try_fused_awq_mlp_streaming,
    )

    reset_awq_runtime_stats()
    x = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
    gate = SimpleNamespace(
        qweight=torch.zeros((64, 4), dtype=torch.int32),
        scales=torch.ones((64, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
        input_size=32,
        output_size=64,
        bias=None,
        prefix="model.layers.0.mlp.gate_proj",
    )
    up = SimpleNamespace(
        qweight=torch.zeros((64, 4), dtype=torch.int32),
        scales=torch.ones((64, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
        input_size=32,
        output_size=64,
        bias=None,
        prefix="model.layers.0.mlp.up_proj",
    )
    down = SimpleNamespace(
        qweight=torch.zeros((32, 8), dtype=torch.int32),
        scales=torch.ones((32, 2), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
        input_size=64,
        output_size=32,
        bias=None,
        prefix="model.layers.0.mlp.down_proj",
    )

    out = try_fused_awq_mlp_streaming(
        x,
        gate,
        up,
        down,
        activation="silu",
        lora_mapping=None,
        inf_config={"kernel_policy": {"awq_mlp_streaming_fusion": True}},
        prefix="model.layers.0.mlp",
    )

    assert out is None
    summary = get_awq_runtime_audit_summary(limit=8)
    assert summary["events"]["mlp_streaming_attempt"] == 1
    assert summary["events"]["mlp_streaming_fallback"] == 1
    assert summary["first_fallbacks"][0]["reason"] == "requires_cross_program_sharing"

def test_packed_int4_down_proj_interleaved_policy_uses_cached_pack(monkeypatch) -> None:
    import torch

    from vllm.model_executor.layers.quantization.tensor import PackedInt4Weight

    reset_awq_runtime_stats()
    qweight = torch.zeros((5376, 2688), dtype=torch.int32)
    scales = torch.ones((5376, 672), dtype=torch.float16)
    weight = PackedInt4Weight(
        qweight,
        scales,
        group_size=32,
        original_shape=(5376, 21504),
        prefix="model.layers.0.mlp.down_proj",
        profile_hint="gemma4_31b_q4",
    )
    x = torch.zeros((1, 21504), dtype=torch.bfloat16)
    expected = torch.full((1, 5376), 0.25, dtype=torch.bfloat16)
    calls = {"pack": 0, "safe": 0}

    def fake_pack(qweight_arg, scales_arg):
        calls["pack"] += 1
        assert qweight_arg is weight.qweight
        assert scales_arg is weight.scales
        return torch.zeros((5376, 672, 5), dtype=torch.int32)

    def fake_safe(a, packed, group_size, *, scale_dtype):
        calls["safe"] += 1
        assert a.shape == (1, 21504)
        assert packed.shape == (5376, 672, 5)
        assert group_size == 32
        assert scale_dtype == weight.scales.dtype
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm."
        "pack_awq_group32_interleaved_qweight_scales",
        fake_pack,
    )
    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm."
        "packed_int4_symmetric_group32_interleaved_gemv_m1_safe",
        fake_safe,
    )

    config = {
        "kernel_policy": {
            "awq_group32_interleaved_down_proj": True,
            "gemma4_dense_down_proj": True,
            "awq_decode_gemv": True,
        }
    }
    y0 = weight.matmul(x, config=config)
    y1 = weight.matmul(x, config=config)

    assert torch.equal(y0, expected)
    assert torch.equal(y1, expected)
    assert calls == {"pack": 1, "safe": 2}
    stats = get_awq_runtime_stats()
    assert stats["awq_interleaved_builds"] == 1
    assert stats["audit:interleaved_down_proj_success"] == 2

def test_gemma4_attention_fused_qkv_helper_splits_outputs(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.models.gemma4.attention import (
        _try_fused_awq_qkv_decode,
    )

    q_proj = SimpleNamespace(
        qweight=torch.zeros((3, 4), dtype=torch.int32),
        scales=torch.ones((3, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
    )
    k_proj = SimpleNamespace(
        qweight=torch.zeros((2, 4), dtype=torch.int32),
        scales=torch.ones((2, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
    )
    v_proj = SimpleNamespace(
        qweight=torch.zeros((2, 4), dtype=torch.int32),
        scales=torch.ones((2, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
    )
    x = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
    fused = torch.arange(7, dtype=torch.bfloat16).reshape(1, 7)

    def fake_safe(*args, **kwargs):
        return fused, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm.packed_int4_symmetric_fused_qkv_m1_safe",
        fake_safe,
    )

    out = _try_fused_awq_qkv_decode(x, q_proj, k_proj, v_proj, inf_config=None)

    assert out is not None
    q, k, v = out
    assert q.shape == (1, 1, 3)
    assert k.shape == (1, 1, 2)
    assert v.shape == (1, 1, 2)
    assert torch.equal(q.reshape(1, 3), fused[:, :3])
    assert torch.equal(k.reshape(1, 2), fused[:, 3:5])
    assert torch.equal(v.reshape(1, 2), fused[:, 5:])

def test_gemma4_attention_fused_qk_helper_reuses_k_as_v(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.models.gemma4.attention import (
        _try_fused_awq_qkv_decode,
    )

    q_proj = SimpleNamespace(
        qweight=torch.zeros((3, 4), dtype=torch.int32),
        scales=torch.ones((3, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
    )
    k_proj = SimpleNamespace(
        qweight=torch.zeros((2, 4), dtype=torch.int32),
        scales=torch.ones((2, 1), dtype=torch.float16),
        qzeros=torch.empty(0),
        group_size=32,
    )
    x = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
    fused = torch.arange(5, dtype=torch.bfloat16).reshape(1, 5)

    def fake_safe(*args, **kwargs):
        return fused, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm.packed_int4_symmetric_fused_qkv_m1_safe",
        fake_safe,
    )

    out = _try_fused_awq_qkv_decode(x, q_proj, k_proj, None, inf_config=None)

    assert out is not None
    q, k, v = out
    assert q.shape == (1, 1, 3)
    assert k.shape == (1, 1, 2)
    assert v.shape == (1, 1, 2)
    assert torch.equal(k, v)
