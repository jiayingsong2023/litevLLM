# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.kernels.triton.paged_attention import _select_paged_attention_launch_config


def test_launch_config_uses_stable_default_for_head256(monkeypatch):
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", raising=False)
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
    )
    assert (warps, stages) == (4, 2)


def test_launch_config_uses_override(monkeypatch):
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", "4")
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", "3")
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
    )
    assert (warps, stages) == (4, 3)


def test_launch_config_uses_scope_override(monkeypatch):
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", raising=False)
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL", "2")
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL", "1")
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="global",
    )
    assert (warps, stages) == (2, 1)


def test_launch_config_scope_beats_global_override(monkeypatch):
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", "4")
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", "2")
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL", "2")
    monkeypatch.setenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL", "2")
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        layer_type="sliding_attention",
    )
    assert (warps, stages) == (2, 2)


def test_launch_config_global_bucket_by_concurrency(monkeypatch):
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL", raising=False)
    warps_low, stages_low = _select_paged_attention_launch_config(
        num_seqs=2,
        head_size=256,
        block_size=16,
        is_int4=False,
        is_fp8=True,
        attn_scope="global",
    )
    warps_high, stages_high = _select_paged_attention_launch_config(
        num_seqs=3,
        head_size=256,
        block_size=16,
        is_int4=False,
        is_fp8=True,
        attn_scope="global",
    )
    assert (warps_low, stages_low) == (2, 2)
    assert (warps_high, stages_high) == (4, 2)


def test_launch_config_local_bucket_by_concurrency(monkeypatch):
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL", raising=False)
    warps_low, stages_low = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="local",
    )
    warps_high, stages_high = _select_paged_attention_launch_config(
        num_seqs=3,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="local",
    )
    assert (warps_low, stages_low) == (4, 1)
    assert (warps_high, stages_high) == (4, 2)


def test_launch_config_c1_preset_single_request(monkeypatch):
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL", raising=False)
    monkeypatch.delenv("FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL", raising=False)
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_C1_PRESET", "1")
    g_w, g_s = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="global",
    )
    l_w, l_s = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="local",
    )
    assert (g_w, g_s) == (4, 1)
    assert (l_w, l_s) == (4, 1)
