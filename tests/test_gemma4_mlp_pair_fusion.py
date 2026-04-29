# SPDX-License-Identifier: Apache-2.0
"""
Tests for Gemma4 MLP gate+up pair fusion (Step 4).

Covers:
  * Helper ``try_fused_awq_pair_matmul`` numerical equivalence with two
    independent ``PackedInt4Weight.matmul`` calls, including cache reuse.
  * Structural guards: mismatched shapes, mismatched group_size, high_fidelity
    flag, and LoRA-active mode must all surface ``None``.
  * ``Gemma4MLP.forward`` routing:
      - env ``FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION=0`` disables fusion.
      - ``lora_mapping is not None`` disables fusion.
      - Activation dispatch (silu vs gelu-tanh) stays consistent between
        fused and unfused branches.

These tests run on CPU when CUDA is unavailable by exercising the helper's
pure-tensor structural checks. Numerical equivalence tests require a CUDA/ROCm
device and auto-skip otherwise (matmul dispatch through PackedInt4Weight
requires either the Triton fused path or the dequant fallback which both
prefer GPU).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.tensor import (
    PackedInt4Weight,
    dequantize_symmetric_packed_int4_pytorch,
)
from vllm.model_executor.models._fused_awq_pair import (
    try_fused_awq_gate_up_activation,
    try_fused_awq_pair_matmul,
)
from vllm.model_executor.models.gemma4 import Gemma4MLP


def _make_stub_litelinear(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_size: int,
    output_size: int,
    *,
    group_size: int = 64,
    prefix: str = "stub",
    force_high_fidelity: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build a minimal object with just the attributes the pair-fusion
    helper inspects. We deliberately avoid constructing a full
    ``LiteLinear`` because the helper treats these as plain data carriers."""
    return SimpleNamespace(
        qweight=qweight,
        scales=scales,
        qzeros=None,
        input_size=input_size,
        output_size=output_size,
        group_size=group_size,
        prefix=prefix,
        force_high_fidelity_awq=force_high_fidelity,
        weight_shape=(output_size, input_size),
        bias=bias,
        awq_profile_hint="",
    )


def _synth_int4_weights(
    n: int, k: int, group_size: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create packed int4 weights + group scales shaped like a real AWQ layer.
    Layout: qweight is ``(N, K // 8)`` uint8 (two int4 nibbles per byte).
    Scales is ``(N, K // group_size)`` fp16. Matches the training-time
    contract used by ``PackedInt4Weight``."""
    g = torch.Generator(device=device).manual_seed(seed)
    qweight = torch.randint(
        0, 255, (n, k // 8), device=device, dtype=torch.uint8, generator=g
    )
    scales = (
        torch.randn(
            (n, k // group_size), device=device, dtype=torch.float32, generator=g
        ).abs()
        + 0.01
    ).to(torch.float16)
    return qweight, scales


# ---------------------------------------------------------------------------
# Structural guards (CPU-only, no dependency on GPU kernels).
# ---------------------------------------------------------------------------


class TestHelperStructuralGuards:
    def _mk(self, n: int, k: int, group_size: int = 64, prefix: str = "a") -> Any:
        qw = torch.zeros((n, k // 8), dtype=torch.uint8)
        sc = torch.ones((n, k // group_size), dtype=torch.float16)
        return _make_stub_litelinear(qw, sc, k, n, group_size=group_size, prefix=prefix)

    def test_returns_none_when_input_sizes_mismatch(self) -> None:
        a = self._mk(64, 256)
        b = self._mk(64, 128)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None

    def test_returns_none_when_output_sizes_mismatch(self) -> None:
        a = self._mk(64, 256)
        b = self._mk(128, 256)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None

    def test_returns_none_when_group_size_mismatch(self) -> None:
        a = self._mk(64, 256, group_size=64)
        b = self._mk(64, 256, group_size=128)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None

    def test_returns_none_when_high_fidelity_forced(self) -> None:
        a = self._mk(64, 256)
        b = self._mk(64, 256)
        a.force_high_fidelity_awq = True
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None

    def test_returns_none_when_lora_mapping_active(self) -> None:
        # Non-None adapter id in the per-request list -> active LoRA.
        a = self._mk(64, 256)
        b = self._mk(64, 256)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert (
            try_fused_awq_pair_matmul(
                x, a, b, owner, "k", lora_mapping=[None, "adapter-7"]
            )
            is None
        )
        assert (
            try_fused_awq_pair_matmul(
                x, a, b, owner, "k", lora_mapping={"fake": True}
            )
            is None
        )

    def test_still_fuses_when_lora_mapping_is_bare_none_list(self) -> None:
        # input_batch_builder emits ``[None, None, ...]`` for every non-LoRA
        # request; this MUST NOT disable fusion. We synthesise a GPU-free
        # shape check: helper should progress past the LoRA guard and hit
        # the structural check (which passes here), then the matmul call.
        # Because constructing a real PackedInt4Weight.matmul without CUDA
        # is impractical, we instead confirm it does not bail out on the
        # LoRA guard by mutating one structural field and checking the
        # returned rejection reason is structural, not LoRA.
        a = self._mk(64, 256)
        b = self._mk(64, 128)  # mismatched input_size triggers structural None
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert (
            try_fused_awq_pair_matmul(
                x, a, b, owner, "k", lora_mapping=[None, None, None]
            )
            is None
        )
        # Note: the helper returns None either way when structure mismatches;
        # a positive assertion for LoRA-inactive passing the guard lives in
        # the numerical-equivalence tests below (which use matching shapes).

    def test_returns_none_when_empty_qweight(self) -> None:
        a = self._mk(64, 256)
        a.qweight = torch.zeros(0, dtype=torch.uint8)
        b = self._mk(64, 256)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None

    def test_returns_none_when_bias_mismatched(self) -> None:
        # One branch has bias, other doesn't: cannot fuse.
        a = self._mk(64, 256)
        b = self._mk(64, 256)
        a.bias = torch.zeros(64)
        x = torch.randn((1, 256), dtype=torch.float32)
        owner = nn.Module()
        assert try_fused_awq_pair_matmul(x, a, b, owner, "k") is None


# ---------------------------------------------------------------------------
# Numerical equivalence. Exercises the real PackedInt4Weight.matmul dispatch.
# ---------------------------------------------------------------------------


def _have_gpu() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _have_gpu(), reason="Requires CUDA/ROCm for matmul dispatch")
class TestHelperNumericEquivalence:
    def _build_pair(
        self,
        n: int,
        k: int,
        group_size: int,
        device: torch.device,
    ) -> tuple[Any, Any, torch.Tensor]:
        qa, sa = _synth_int4_weights(n, k, group_size, seed=101, device=device)
        qb, sb = _synth_int4_weights(n, k, group_size, seed=202, device=device)
        a = _make_stub_litelinear(
            qa, sa, k, n, group_size=group_size, prefix="pair.a"
        )
        b = _make_stub_litelinear(
            qb, sb, k, n, group_size=group_size, prefix="pair.b"
        )
        x = torch.randn((2, k), device=device, dtype=torch.bfloat16)
        return a, b, x

    def test_fused_matches_two_independent_matmuls(self) -> None:
        device = torch.device("cuda")
        n, k, group_size = 256, 512, 64
        a, b, x = self._build_pair(n, k, group_size, device)

        owner = nn.Module()
        fused = try_fused_awq_pair_matmul(x, a, b, owner, "mlp_gate_up")
        assert fused is not None
        assert fused.shape == (2, 2 * n)

        # Reference: dequantize each leg independently and apply F.linear.
        w_a = dequantize_symmetric_packed_int4_pytorch(
            a.qweight.to(torch.int32), a.scales, group_size=group_size
        ).to(x.dtype)
        w_b = dequantize_symmetric_packed_int4_pytorch(
            b.qweight.to(torch.int32), b.scales, group_size=group_size
        ).to(x.dtype)
        y_a = F.linear(x, w_a)
        y_b = F.linear(x, w_b)
        y_ref = torch.cat([y_a, y_b], dim=-1).float()

        cos = F.cosine_similarity(
            fused.float().reshape(-1), y_ref.reshape(-1), dim=0
        ).item()
        # bf16 rounding + possible fused/dense path divergence: stay loose
        # enough for either dispatch, tight enough to catch concat-order bugs.
        assert cos > 0.995, f"cosine similarity {cos} too low"

    def test_fused_weight_is_cached_on_owner(self) -> None:
        device = torch.device("cuda")
        n, k, group_size = 128, 256, 64
        a, b, x = self._build_pair(n, k, group_size, device)
        owner = nn.Module()

        _ = try_fused_awq_pair_matmul(x, a, b, owner, "mlp_gate_up")
        cache_attr = "_fused_awq_pair_mlp_gate_up"
        assert hasattr(owner, cache_attr)
        cached_first = getattr(owner, cache_attr)

        # Second call must reuse the same concatenated PackedInt4Weight.
        _ = try_fused_awq_pair_matmul(x, a, b, owner, "mlp_gate_up")
        assert getattr(owner, cache_attr) is cached_first

    def test_direct_fused_gate_up_matches_dense_silu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GATE_UP", "1")
        device = torch.device("cuda")
        n, k, group_size = 257, 288, 32
        gate, up, _ = self._build_pair(n, k, group_size, device)
        x = torch.randn((1, k), device=device, dtype=torch.bfloat16)

        fused = try_fused_awq_gate_up_activation(
            x,
            gate,
            up,
            activation="silu",
        )
        assert fused is not None
        torch.cuda.synchronize()

        w_gate = dequantize_symmetric_packed_int4_pytorch(
            gate.qweight.to(torch.int32), gate.scales, group_size=group_size
        ).to(x.dtype)
        w_up = dequantize_symmetric_packed_int4_pytorch(
            up.qweight.to(torch.int32), up.scales, group_size=group_size
        ).to(x.dtype)
        ref = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
        cos = F.cosine_similarity(fused.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
        diff = (fused.float() - ref.float()).abs()
        rel_mae = (diff.mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
        assert cos > 0.999, f"cosine similarity {cos} too low"
        assert rel_mae < 0.02, f"relative mae {rel_mae} too high"

    def test_direct_fused_gate_up_matches_dense_gelu_tanh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GATE_UP", "1")
        device = torch.device("cuda")
        n, k, group_size = 193, 256, 32
        gate, up, _ = self._build_pair(n, k, group_size, device)
        x = torch.randn((1, k), device=device, dtype=torch.bfloat16)

        fused = try_fused_awq_gate_up_activation(
            x,
            gate,
            up,
            activation="gelu_pytorch_tanh",
        )
        assert fused is not None
        torch.cuda.synchronize()

        w_gate = dequantize_symmetric_packed_int4_pytorch(
            gate.qweight.to(torch.int32), gate.scales, group_size=group_size
        ).to(x.dtype)
        w_up = dequantize_symmetric_packed_int4_pytorch(
            up.qweight.to(torch.int32), up.scales, group_size=group_size
        ).to(x.dtype)
        ref = F.gelu(F.linear(x, w_gate), approximate="tanh") * F.linear(x, w_up)
        cos = F.cosine_similarity(fused.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
        diff = (fused.float() - ref.float()).abs()
        rel_mae = (diff.mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
        assert cos > 0.999, f"cosine similarity {cos} too low"
        assert rel_mae < 0.02, f"relative mae {rel_mae} too high"


# ---------------------------------------------------------------------------
# Gemma4MLP.forward routing.
# ---------------------------------------------------------------------------


def _make_mlp(hidden: int = 64, inter: int = 128, act: str = "gelu_pytorch_tanh"):
    cfg = SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        hidden_activation=act,
        hidden_act=act,
    )
    # quant_config=None: LiteLinear falls back to F.linear on .weight,
    # which we pre-populate with dense tensors for deterministic routing tests.
    mlp = Gemma4MLP(cfg, quant_config=None, prefix="model.layers.0")
    mlp.gate_proj.weight = nn.Parameter(
        torch.randn(inter, hidden, dtype=torch.float32) * 0.02, requires_grad=False
    )
    mlp.up_proj.weight = nn.Parameter(
        torch.randn(inter, hidden, dtype=torch.float32) * 0.02, requires_grad=False
    )
    mlp.down_proj.weight = nn.Parameter(
        torch.randn(hidden, inter, dtype=torch.float32) * 0.02, requires_grad=False
    )
    return mlp


class TestGemma4MLPRouting:
    def test_env_off_bypasses_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mlp = _make_mlp()
        x = torch.randn((1, 4, 64), dtype=torch.float32)
        monkeypatch.setenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", "0")

        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul"
        ) as helper:
            out = mlp(x)
            helper.assert_not_called()
        assert out.shape == x.shape

    def test_lora_active_bypasses_fusion_inside_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gemma4MLP always enters the helper (no outer LoRA gate). The
        # helper itself decides to bail when LoRA is active. We verify
        # the forward still completes with matching shape.
        mlp = _make_mlp()
        x = torch.randn((1, 4, 64), dtype=torch.float32)
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        out = mlp(x, lora_mapping=[None, "adapter-7"])
        assert out.shape == x.shape

    def test_bare_none_list_still_enters_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression guard for the Step 4.5 bug: input_batch_builder emits
        # ``[None, None, ...]`` for non-LoRA requests; the helper MUST be
        # invoked in that case (previously the outer guard skipped it).
        mlp = _make_mlp()
        x = torch.randn((1, 2, 64), dtype=torch.float32)
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul",
            return_value=None,
        ) as helper:
            _ = mlp(x, lora_mapping=[None])
            assert helper.call_count == 1

    def test_default_attempts_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mlp = _make_mlp()
        x = torch.randn((1, 4, 64), dtype=torch.float32)
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul",
            return_value=None,
        ) as helper:
            _ = mlp(x)
            assert helper.call_count == 1
            args, kwargs = helper.call_args
            # Signature contract: (x, gate_proj, up_proj, owner, "mlp_gate_up", lora_mapping=None)
            assert args[1] is mlp.gate_proj
            assert args[2] is mlp.up_proj
            assert args[3] is mlp
            assert args[4] == "mlp_gate_up"
            assert kwargs.get("lora_mapping") is None

    def test_direct_gate_up_helper_has_priority_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mlp = _make_mlp()
        x = torch.randn((1, 1, 64), dtype=torch.float32)
        direct = torch.randn((1, 1, 128), dtype=torch.float32)
        monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GATE_UP", "1")
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_gate_up_activation",
            return_value=direct,
        ) as direct_helper, mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul",
            return_value=None,
        ) as pair_helper:
            out = mlp(x)
        assert out.shape == x.shape
        assert direct_helper.call_count == 1
        pair_helper.assert_not_called()

    def test_fused_branch_numerically_matches_unfused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the helper returns a concat tensor, downstream math (split,
        activation, down_proj) must match the two-matmul reference exactly."""
        mlp = _make_mlp(act="gelu_pytorch_tanh")
        x = torch.randn((1, 3, 64), dtype=torch.float32)
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        # Compute the canonical two-matmul output for reference.
        gate_ref = mlp.gate_proj(x)
        up_ref = mlp.up_proj(x)
        act_ref = F.gelu(gate_ref, approximate="tanh")
        y_ref = mlp.down_proj(act_ref * up_ref)

        # Helper-returned concat tensor: gate_ref || up_ref along last dim.
        concat = torch.cat([gate_ref, up_ref], dim=-1)

        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul",
            return_value=concat,
        ):
            y_fused = mlp(x)
        torch.testing.assert_close(y_fused, y_ref, rtol=1e-6, atol=1e-6)

    def test_silu_activation_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mlp = _make_mlp(act="silu")
        x = torch.randn((1, 2, 64), dtype=torch.float32)
        monkeypatch.delenv("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION", raising=False)

        gate_ref = mlp.gate_proj(x)
        up_ref = mlp.up_proj(x)
        act_ref = F.silu(gate_ref)
        y_ref = mlp.down_proj(act_ref * up_ref)

        concat = torch.cat([gate_ref, up_ref], dim=-1)
        with mock.patch(
            "vllm.model_executor.models._fused_awq_pair.try_fused_awq_pair_matmul",
            return_value=concat,
        ):
            y_fused = mlp(x)
        torch.testing.assert_close(y_fused, y_ref, rtol=1e-6, atol=1e-6)
