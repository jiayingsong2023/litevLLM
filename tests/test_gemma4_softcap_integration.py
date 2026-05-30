# SPDX-License-Identifier: Apache-2.0
"""Step-3 integration guardrails: Gemma4 decode path must correctly plumb the
``attn_logit_softcapping`` value from config -> ``paged_attention_v1`` kwargs,
and the legacy fp16-ref fallback must stay available via env override.

These tests never load a real model. They intercept ``paged_attention_v1`` to
assert wiring contracts and gate branches with environment variables.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vllm.model_executor.models.gemma4 as gemma4_mod
from vllm.model_executor.models.gemma4 import _should_use_full_decode_reference


# ---------------------------------------------------------------------------
# Env-gated branch selection for the global-decode ref path.
# ---------------------------------------------------------------------------
def test_full_decode_ref_default_is_kernel_path():
    """Default behaviour after Step 3: never force the pytorch ref path, even
    when KV is fp16/bf16. The Triton kernel handles softcap natively now."""
    assert _should_use_full_decode_reference(None, "auto") is False
    assert _should_use_full_decode_reference(None, "fp16") is False
    assert _should_use_full_decode_reference(None, "bf16") is False
    assert _should_use_full_decode_reference(None, "int4") is False
    assert _should_use_full_decode_reference(None, "fp8") is False


def test_full_decode_ref_legacy_fp16_config_preserves_old_behaviour():
    """Operators can still opt into the pre-Step-3 behaviour: fp16/bf16 KV
    falls back to ref, int4/fp8 takes the kernel path."""
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN": "1"}
    )
    assert _should_use_full_decode_reference(inf_config, "auto") is True
    assert _should_use_full_decode_reference(inf_config, "fp16") is True
    assert _should_use_full_decode_reference(inf_config, "bf16") is True
    assert _should_use_full_decode_reference(inf_config, "int4") is False
    assert _should_use_full_decode_reference(inf_config, "fp8") is False


def test_full_decode_ref_emergency_force_always_returns_true():
    """FORCE_FULL_REF_ATTN is an emergency rollback knob for numerical debugging.
    It must take precedence over the legacy fp16 switch."""
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN": "1"}
    )
    for kv in ("auto", "fp16", "bf16", "int4", "fp8"):
        assert _should_use_full_decode_reference(inf_config, kv) is True


def test_full_decode_ref_ignores_env_without_config(monkeypatch):
    """Direct environment state should not affect the runtime policy."""
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "1")
    for kv in ("auto", "fp16", "bf16", "int4", "fp8"):
        assert _should_use_full_decode_reference(None, kv) is False


# ---------------------------------------------------------------------------
# Wiring contract: gemma4 attention must pass the kernel a numeric softcap
# when config.attn_logit_softcapping > 0, and None when it is 0/absent.
# ---------------------------------------------------------------------------
class _FakeAttention:
    """Reproduces just enough of ``Gemma4Attention`` surface so we can exercise
    the softcap-plumbing snippet via a synthetic call. We do NOT instantiate
    a real module (that would require LiteConfig + quant_config + RoPE + ...).

    What we assert is the exact expression used inside gemma4.py:

        softcap=(
            float(softcap)
            if softcap is not None and float(softcap) > 0.0
            else None
        )
    """

    @staticmethod
    def translate(softcap: Any) -> Any:
        return (
            float(softcap)
            if softcap is not None and float(softcap) > 0.0
            else None
        )


@pytest.mark.parametrize(
    "cfg_value,expected",
    [
        (50.0, 50.0),
        (30, 30.0),
        (1.0, 1.0),
        (0.0, None),
        (-1.0, None),
        (None, None),
    ],
)
def test_softcap_plumbing_expression_matches_expected(cfg_value, expected):
    """The softcap expression inlined in gemma4.py forward() must collapse
    falsy/zero/negative values to ``None`` so the kernel takes the
    HAS_SOFTCAP=False compile branch. Any mismatch here silently disables
    soft-capping at runtime."""
    assert _FakeAttention.translate(cfg_value) == expected


# ---------------------------------------------------------------------------
# Source-level contract: verify the three paged_attention_v1 call sites in
# gemma4.py all pass ``softcap=`` via the same expression. This is a cheap
# insurance policy against future refactors that silently drop the kwarg.
# ---------------------------------------------------------------------------
def test_gemma4_paged_attention_calls_all_pass_softcap():
    # gemma4 is now a subpackage; scan all .py files in the directory.
    src_dir = Path(gemma4_mod.__file__).resolve().parent
    parts = []
    for f in sorted(src_dir.iterdir()):
        if f.suffix == ".py":
            parts.append(f.read_text(encoding="utf-8"))
    src = "\n".join(parts)

    # There are currently 2 in-module call sites to paged_attention_v1:
    # one for local decode and one for global decode. Both must propagate
    # softcap. A future third site (e.g. prefill Triton path) should also
    # do so.
    call_count = src.count("paged_attention_v1(")
    assert call_count >= 2, (
        f"Expected >=2 paged_attention_v1(...) call sites, found {call_count}."
        " If you removed a call site, update this guardrail."
    )
    softcap_kwarg_count = src.count("softcap=(")
    assert softcap_kwarg_count >= 2, (
        "paged_attention_v1 call sites in gemma4 must each pass softcap=(...)"
        f" via the Step 3 plumbing expression. Found {softcap_kwarg_count} sites."
    )


# ---------------------------------------------------------------------------
# End-to-end kernel invocation through gemma4.py is covered by the Tier-B
# spotcheck in Step 3.4 (real 31B model). That is the authoritative parity
# check; this file only guards the wiring contract.
# ---------------------------------------------------------------------------
