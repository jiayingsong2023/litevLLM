# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from tests import e2e_full_benchmark as bench


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        warmup_prefill_rounds=1,
        warmup_decode_rounds=2,
        warmup_decode_tokens=8,
        warmup_burst_rounds=0,
        warmup_burst_concurrency=0,
        warmup_burst_decode_tokens=8,
    )


def test_resolve_warmup_config_from_args_defaults(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_BURST_ROUNDS", raising=False)
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_BURST_CONCURRENCY", raising=False)
    monkeypatch.delenv("FASTINFERENCE_BENCH_WARMUP_BURST_DECODE_TOKENS", raising=False)

    cfg = bench._resolve_warmup_config(_args())
    assert cfg.prefill_rounds == 1
    assert cfg.decode_rounds == 2
    assert cfg.decode_tokens == 8
    assert cfg.burst_rounds == 0
    assert cfg.burst_concurrency == 0
    assert cfg.burst_decode_tokens == 8


def test_resolve_warmup_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS", "3")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS", "4")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS", "12")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_BURST_ROUNDS", "2")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_BURST_CONCURRENCY", "3")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_BURST_DECODE_TOKENS", "9")

    cfg = bench._resolve_warmup_config(_args())
    assert cfg.prefill_rounds == 3
    assert cfg.decode_rounds == 4
    assert cfg.decode_tokens == 12
    assert cfg.burst_rounds == 2
    assert cfg.burst_concurrency == 3
    assert cfg.burst_decode_tokens == 9


def test_resolve_warmup_config_invalid_env_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS", "bad")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS", "bad")
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS", "bad")
    cfg = bench._resolve_warmup_config(_args())
    assert cfg.prefill_rounds == 1
    assert cfg.decode_rounds == 2
    assert cfg.decode_tokens == 8
