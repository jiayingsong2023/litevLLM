# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.kernels.triton.paged_attention import _select_paged_attention_launch_config


def test_launch_config_uses_stable_default_for_head256():
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
    )
    assert (warps, stages) == (4, 2)


def test_launch_config_uses_override():
    cfg = SimpleNamespace(paged_attn_num_warps=4, paged_attn_num_stages=3)
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        config=cfg,
    )
    assert (warps, stages) == (4, 3)


def test_launch_config_uses_scope_override():
    cfg = SimpleNamespace(
        paged_attn_num_warps_global=2,
        paged_attn_num_stages_global=1,
    )
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="global",
        config=cfg,
    )
    assert (warps, stages) == (2, 1)


def test_launch_config_scope_beats_global_override():
    cfg = SimpleNamespace(
        paged_attn_num_warps=4,
        paged_attn_num_stages=2,
        paged_attn_num_warps_local=2,
        paged_attn_num_stages_local=2,
    )
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        layer_type="sliding_attention",
        config=cfg,
    )
    assert (warps, stages) == (2, 2)


def test_launch_config_global_bucket_by_concurrency():
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


def test_launch_config_gemma4_31b_c1_global_bucket_uses_stage1():
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=512,
        block_size=16,
        is_int4=False,
        is_fp8=True,
        attn_scope="global",
    )
    assert (warps, stages) == (4, 1)


def test_launch_config_local_bucket_by_concurrency():
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


def test_launch_config_c1_preset_single_request():
    cfg = SimpleNamespace(gemma4_c1_preset=True)
    g_w, g_s = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="global",
        config=cfg,
    )
    l_w, l_s = _select_paged_attention_launch_config(
        num_seqs=1,
        head_size=256,
        block_size=16,
        is_int4=True,
        is_fp8=False,
        attn_scope="local",
        config=cfg,
    )
    assert (g_w, g_s) == (4, 1)
    assert (l_w, l_s) == (4, 1)
