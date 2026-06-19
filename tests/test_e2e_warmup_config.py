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
    args = argparse.Namespace(
        compile_cache_dir=str(tmp_path / "compile_cache"), compile_cache_clear=False
    )
    env_map, meta = bench._resolve_compile_cache_env(args)
    assert meta["enabled"] is True
    assert "TRITON_CACHE_DIR" in env_map
    assert "TORCHINDUCTOR_CACHE_DIR" in env_map


def test_deepseek_v4_flash_gguf_is_registered_for_e2e_benchmark() -> None:
    spec = bench.MODEL_SPECS["deepseek_v4_flash_q2_gguf"]

    assert spec.model_path.endswith(".gguf")
    assert spec.quant == "deepseek-v4-flash-gguf"
    assert spec.concurrent_reqs == 1
    assert spec.prompt_tokens_target == 4096
    assert spec.max_new_tokens == 16


def test_deepseek_v4_flash_gguf_is_in_default_e2e_models(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["e2e_full_benchmark.py"])

    args = bench._parse_args()

    assert "deepseek_v4_flash_q2_gguf" in {
        key.strip() for key in args.models.split(",")
    }


def test_runtime_metrics_include_async_driver_snapshot() -> None:
    metrics = bench._derive_runtime_metrics(
        {
            "observer": {"step_count": 8},
            "backend": {},
            "async_driver": {
                "steps": 10,
                "backpressure_sleeps": 7,
                "idle_waits": 2,
                "background_errors": 1,
                "min_step_interval_s": 0.003,
            },
        }
    )

    async_metrics = metrics["async_driver"]
    assert async_metrics["steps"] == 10.0
    assert async_metrics["backpressure_sleep_rate"] == 0.7
    assert async_metrics["background_error_rate"] == 0.1
    assert async_metrics["observer_step_gap"] == 2.0


def test_runtime_phase_diffs_include_async_driver_delta() -> None:
    diffs = bench._derive_runtime_phase_diffs(
        {
            "warmup": {
                "derived_metrics": {
                    "async_driver": {
                        "steps": 2,
                        "backpressure_sleeps": 1,
                        "idle_waits": 1,
                        "background_errors": 0,
                        "backpressure_sleep_rate": 0.5,
                    }
                }
            },
            "benchmark": {
                "derived_metrics": {
                    "async_driver": {
                        "steps": 7,
                        "backpressure_sleeps": 6,
                        "idle_waits": 1,
                        "background_errors": 1,
                        "backpressure_sleep_rate": 0.857,
                    }
                }
            },
        }
    )

    async_delta = diffs["async_driver"]["benchmark_delta"]
    assert async_delta["steps"] == 5.0
    assert async_delta["backpressure_sleeps"] == 5.0
    assert async_delta["background_errors"] == 1.0
    assert async_delta["backpressure_sleep_rate"] > 0.35
    assert any(
        line.startswith("  RUNTIME(async):")
        for line in bench._format_runtime_phase_diff_summary(diffs)
    )


def test_evaluate_perf_regressions_warns_without_failing() -> None:
    warnings = bench._evaluate_perf_regressions(
        {
            "gemma4_31b_q4": {
                "aggregate_tps": 8.0,
                "prefill_tps_aggregate": 100.0,
                "decode_tps_aggregate": 4.0,
                "ttft_p50_ms": 130.0,
            }
        },
        {
            "gemma4_31b_q4": {
                "aggregate_tps": 10.0,
                "prefill_tps_aggregate": 100.0,
                "decode_tps_aggregate": 5.0,
                "ttft_p50_ms": 100.0,
            }
        },
        min_tps_ratio=0.85,
        max_latency_ratio=1.25,
    )

    assert {item["metric"] for item in warnings} == {
        "aggregate_tps",
        "decode_tps_aggregate",
        "ttft_p50_ms",
    }
    assert all("model" in item for item in warnings)
    assert all(
        line.startswith("PERF WARNING:")
        for line in bench._format_perf_regression_warnings(warnings)
    )
