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


def test_benchmark_fingerprint_records_comparison_inputs(monkeypatch) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    args = argparse.Namespace(workload="text", fixed_decode_len=True, warmup_preset="off")

    assert bench._benchmark_fingerprint(args, ["tinyllama"]) == {
        "models": ["tinyllama"],
        "gpu": None,
        "gpu_arch": None,
        "rocm": bench.torch.version.hip,
        "torch": bench.torch.__version__,
        "triton": getattr(bench.torch.version, "triton", None),
        "workload": "text",
        "fixed_decode_len": True,
        "warmup_preset": "off",
        "shape": {},
        "runtime_env": {
            "FASTINFERENCE_KV_TYPE": None,
            "FASTINFERENCE_FUSION_LEVEL": None,
        },
    }


def test_deepseek_v4_flash_gguf_is_registered_for_e2e_benchmark() -> None:
    spec = bench.MODEL_SPECS["deepseek_v4_flash_q2_gguf"]

    assert spec.model_path.endswith(".gguf")
    assert spec.quant == "deepseek-v4-flash-gguf"
    assert spec.concurrent_reqs == 1
    assert spec.prompt_tokens_target == 32
    assert spec.max_new_tokens == 16


def test_gemma4_26b_does_not_inject_legacy_runtime_env() -> None:
    spec = bench.MODEL_SPECS["gemma4_26b_a4b"]

    assert spec.stable_env == {}


def test_gemma4_31b_does_not_inject_legacy_runtime_env() -> None:
    spec = bench.MODEL_SPECS["gemma4_31b_q4"]

    assert spec.stable_env == {}


def test_deepseek_v4_flash_gguf_benchmark_requests_full_resident(monkeypatch) -> None:
    captured_command: list[str] = []

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = (
            '{"context_length":4096,"max_tokens":1,"repeat":1,'
            '"runs":[{"elapsed_ms":1000.0,"generated_token_count":1,'
            '"decode_tokens_total":1,"decode_ms_total":1000.0,'
            '"decode_tps_steady_state":1.0}],"gpu_backend":{},'
            '"gpu_staging":{},"runtime_budget":{}}'
        )

    def fake_run(*_args, **kwargs):
        captured_command.extend(_args[0])
        return FakeProc()

    monkeypatch.setattr(bench.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    result = bench._run_deepseek_v4_flash_gguf_benchmark(
        bench.MODEL_SPECS["deepseek_v4_flash_q2_gguf"]
    )

    assert result["skipped"] == 0.0
    assert "--full-resident" in captured_command


def test_deepseek_v4_flash_gguf_is_in_default_e2e_models(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["e2e_full_benchmark.py"])

    args = bench._parse_args()

    assert "deepseek_v4_flash_q2_gguf" in {
        key.strip() for key in args.models.split(",")
    }


def test_deepseek_v4_flash_decode_p50_target_warning_output() -> None:
    warning = bench._deepseek_v4_flash_decode_target_warning(
        {
            "deepseek_v4_flash_q2_gguf": {
                "decode_tps_aggregate": 1.2,
                "decode_tps_p50": 0.83,
            }
        }
    )

    assert warning == {
        "model": "deepseek_v4_flash_q2_gguf",
        "metric": "decode_tps_p50",
        "kind": "throughput_target",
        "current": 0.83,
        "baseline": 1.0,
        "ratio": 0.83,
        "threshold": 1.0,
    }
    assert bench._format_perf_regression_warnings([warning]) == [
        "PERF WARNING: model=deepseek_v4_flash_q2_gguf "
        "metric=decode_tps_p50 kind=throughput_target current=0.830 "
        "baseline=1.000 ratio=0.830 threshold=1.000"
    ]


def test_deepseek_v4_flash_missing_decode_p50_warning_output() -> None:
    for result in (
        {"decode_tps_aggregate": 1.2},
        {"decode_tps_aggregate": 1.2, "decode_tps_p50": float("nan")},
    ):
        warning = bench._deepseek_v4_flash_decode_target_warning(
            {"deepseek_v4_flash_q2_gguf": result}
        )

        assert warning is not None
        assert warning["kind"] == "throughput_target_missing"
        assert bench._format_perf_regression_warnings([warning]) == [
            "PERF WARNING: model=deepseek_v4_flash_q2_gguf "
            "metric=decode_tps_p50 kind=throughput_target_missing current=n/a "
            "baseline=1.000 ratio=n/a threshold=1.000"
        ]


def test_deepseek_v4_flash_decode_p50_target_accepts_threshold() -> None:
    assert (
        bench._deepseek_v4_flash_decode_target_warning(
            {
                "deepseek_v4_flash_q2_gguf": {
                    "decode_tps_p50": 1.0,
                },
                "tinyllama": {"decode_tps_p50": 0.1},
            }
        )
        is None
    )


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


def test_perf_gate_is_opt_in() -> None:
    regressions = [{"metric": "aggregate_tps"}]

    assert not bench._perf_gate_failed(regressions, fail_on_regression=False)
    assert bench._perf_gate_failed(regressions, fail_on_regression=True)
