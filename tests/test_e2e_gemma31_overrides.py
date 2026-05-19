# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from types import SimpleNamespace

from tests import e2e_full_benchmark as bench


def test_parse_args_defaults_to_gemma4_26b_and_31b(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["e2e_full_benchmark.py"])
    args = bench._parse_args()
    assert args.models == "gemma4_26b_a4b,gemma4_31b_q4"


def test_format_targets_uses_model_display_names() -> None:
    specs = [
        bench.MODEL_SPECS["gemma4_26b_a4b"],
        bench.MODEL_SPECS["gemma4_31b_q4"],
    ]
    assert bench._format_targets(specs) == "Gemma4-26B A4B (Q4 MoE) + Gemma4-31B (Q4)"


def test_parse_args_accepts_gemma31_shape_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_full_benchmark.py",
            "--models",
            "gemma4_31b_q4",
            "--gemma31b-prompt-tokens",
            "256",
            "--gemma31b-max-new-tokens",
            "64",
            "--gemma31b-max-model-len",
            "768",
        ],
    )
    args = bench._parse_args()
    assert args.models == "gemma4_31b_q4"
    assert args.gemma31b_prompt_tokens == 256
    assert args.gemma31b_max_new_tokens == 64
    assert args.gemma31b_max_model_len == 768


def test_parse_args_accepts_gemma31_bucket_policy_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_full_benchmark.py",
            "--models",
            "gemma4_31b_q4",
            "--gemma31b-bucket-disable",
            "--gemma31b-bucket-cutoff",
            "128",
            "--gemma31b-bucket-decode-cutoff",
            "32",
            "--gemma31b-schedule-profile",
            "baseline",
        ],
    )
    args = bench._parse_args()
    assert args.gemma31b_bucket_enable is False
    assert args.gemma31b_bucket_cutoff == 128
    assert args.gemma31b_bucket_decode_cutoff == 32
    assert args.gemma31b_schedule_profile == "baseline"


def test_resolve_gemma31_bucket_policy_short_prompt_short_decode_uses_catchup_prefill(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_ENABLE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_SCHED_PROFILE", raising=False)
    spec = replace(
        bench.MODEL_SPECS["gemma4_31b_q4"],
        prompt_tokens_target=128,
        max_new_tokens=32,
    )
    args = argparse.Namespace(
        gemma31b_bucket_enable=True,
        gemma31b_bucket_cutoff=128,
        gemma31b_bucket_decode_cutoff=32,
        gemma31b_bucket_short_profile="decode_bias",
        gemma31b_bucket_short_decode_profile="catchup_prefill",
        gemma31b_bucket_long_profile="baseline",
        gemma31b_schedule_profile="auto_bucket",
    )
    updated, policy = bench._resolve_gemma31b_bucket_policy(spec, args)
    assert policy["selected_profile"] == "catchup_prefill"
    assert updated.stable_env["FASTINFERENCE_LITE_PREFILL_CHUNK"] == "384"


def test_resolve_gemma31_bucket_policy_short_prompt_long_decode_uses_decode_bias(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_ENABLE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_SCHED_PROFILE", raising=False)
    spec = replace(
        bench.MODEL_SPECS["gemma4_31b_q4"],
        prompt_tokens_target=128,
        max_new_tokens=64,
    )
    args = argparse.Namespace(
        gemma31b_bucket_enable=True,
        gemma31b_bucket_cutoff=128,
        gemma31b_bucket_decode_cutoff=32,
        gemma31b_bucket_short_profile="decode_bias",
        gemma31b_bucket_short_decode_profile="catchup_prefill",
        gemma31b_bucket_long_profile="baseline",
        gemma31b_schedule_profile="auto_bucket",
    )
    updated, policy = bench._resolve_gemma31b_bucket_policy(spec, args)
    assert policy["selected_profile"] == "decode_bias"
    assert updated.stable_env["FASTINFERENCE_LITE_PREFILL_CHUNK"] == "192"


def test_resolve_gemma31_bucket_policy_long_prompt_uses_baseline(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_ENABLE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF", raising=False)
    monkeypatch.delenv("FASTINFERENCE_GEMMA31B_SCHED_PROFILE", raising=False)
    spec = replace(
        bench.MODEL_SPECS["gemma4_31b_q4"],
        prompt_tokens_target=384,
        max_new_tokens=32,
    )
    args = argparse.Namespace(
        gemma31b_bucket_enable=True,
        gemma31b_bucket_cutoff=128,
        gemma31b_bucket_decode_cutoff=32,
        gemma31b_bucket_short_profile="decode_bias",
        gemma31b_bucket_short_decode_profile="catchup_prefill",
        gemma31b_bucket_long_profile="baseline",
        gemma31b_schedule_profile="auto_bucket",
    )
    updated, policy = bench._resolve_gemma31b_bucket_policy(spec, args)
    assert policy["selected_profile"] == "baseline"
    assert updated.stable_env["FASTINFERENCE_LITE_PREFILL_CHUNK"] == "256"


def test_gemma4_model_specs_include_recommended_profiles() -> None:
    gemma31 = bench.MODEL_SPECS["gemma4_31b_q4"]
    gemma26 = bench.MODEL_SPECS["gemma4_26b_a4b"]

    assert gemma31.stable_env["FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ"] == "1"
    assert gemma31.stable_env["FASTINFERENCE_AWQ_DECODE_GEMV"] == "1"
    assert gemma31.stable_env["FASTINFERENCE_AWQ_GROUP32_GEMV_ALL"] == "1"
    assert gemma31.stable_env["FASTINFERENCE_AWQ_FUSED_GATE_UP"] == "1"
    assert gemma31.stable_env["FASTINFERENCE_GPU_GREEDY_SAMPLING"] == "1"

    assert gemma26.stable_env["FASTINFERENCE_AWQ_DECODE_GEMV"] == "1"
    assert gemma26.stable_env["FASTINFERENCE_AWQ_FUSED_GATE_UP"] == "1"
    assert gemma26.stable_env["FASTINFERENCE_GPU_GREEDY_SAMPLING"] == "1"
    assert gemma26.stable_env["FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE"] == "32"
    assert gemma26.stable_env["FASTINFERENCE_GEMMA4_MOE_COMPUTE_DTYPE"] == "auto"
    assert gemma26.stable_env["FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL"] == "1"
    assert "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ" not in gemma26.stable_env


def test_gemma4_adapter_installs_benchmark_recommended_tuning_defaults() -> None:
    from vllm.adapters.gemma4 import Gemma4Adapter

    adapter = Gemma4Adapter()

    dense_policy = adapter.runtime_policy(
        SimpleNamespace(hf_config=SimpleNamespace()),
        SimpleNamespace(tuning_env={}),
    )
    dense_env = dense_policy.tuning_env_overrides
    assert dense_env["FASTINFERENCE_AWQ_DECODE_GEMV"] == "1"
    assert dense_env["FASTINFERENCE_AWQ_FUSED_GATE_UP"] == "1"
    assert dense_env["FASTINFERENCE_AWQ_GROUP32_GEMV_ALL"] == "1"
    assert dense_env["FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ"] == "1"

    moe_policy = adapter.runtime_policy(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=1408,
            )
        ),
        SimpleNamespace(tuning_env={}),
    )
    moe_env = moe_policy.tuning_env_overrides
    assert moe_env["FASTINFERENCE_AWQ_DECODE_GEMV"] == "1"
    assert moe_env["FASTINFERENCE_AWQ_FUSED_GATE_UP"] == "1"
    assert "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ" not in moe_env


def test_should_auto_isolate_multi_gemma_on_rocm(monkeypatch) -> None:
    monkeypatch.setattr(bench.torch.version, "hip", "6.1", raising=False)
    args = argparse.Namespace(
        model_process_isolation=False,
        no_model_process_isolation=False,
    )
    assert bench._should_use_model_process_isolation(
        args,
        ["gemma4_26b_a4b", "gemma4_31b_q4"],
    )


def test_should_not_auto_isolate_single_model(monkeypatch) -> None:
    monkeypatch.setattr(bench.torch.version, "hip", "6.1", raising=False)
    args = argparse.Namespace(
        model_process_isolation=False,
        no_model_process_isolation=False,
    )
    assert not bench._should_use_model_process_isolation(
        args,
        ["gemma4_31b_q4"],
    )
