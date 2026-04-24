# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

from tests import e2e_full_benchmark as bench


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        warmup_preset="default",
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


def test_resolve_warmup_config_off_preset(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_PRESET", "off")
    cfg = bench._resolve_warmup_config(_args())
    assert cfg.prefill_rounds == 0
    assert cfg.decode_rounds == 0
    assert cfg.burst_rounds == 0


def test_resolve_warmup_config_cold_preset(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_BENCH_WARMUP_PRESET", "cold")
    cfg = bench._resolve_warmup_config(_args())
    assert cfg.prefill_rounds == 2
    assert cfg.decode_rounds == 2
    assert cfg.decode_tokens == 16
    assert cfg.burst_rounds == 1
    assert cfg.burst_concurrency == 2


def test_resolve_compile_cache_env(tmp_path: Path) -> None:
    args = argparse.Namespace(compile_cache_dir=str(tmp_path / "compile_cache"), compile_cache_clear=False)
    env_map, meta = bench._resolve_compile_cache_env(args)
    assert meta["enabled"] is True
    assert "TRITON_CACHE_DIR" in env_map
    assert "TORCHINDUCTOR_CACHE_DIR" in env_map
