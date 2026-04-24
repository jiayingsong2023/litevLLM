# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import argparse
from dataclasses import replace

from tests import e2e_full_benchmark as bench


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
