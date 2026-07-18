# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _tool_module():
    path = Path(__file__).parent / "tools" / "interleaved_e2e_ab.py"
    spec = importlib.util.spec_from_file_location("interleaved_e2e_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(*, fingerprint: dict[str, object]) -> dict[str, object]:
    return {
        "fingerprint": fingerprint,
        "summary": {
            "model-a": {"decode_tps_aggregate": 10.0, "profile": {"kv": "fp8"}},
            "model-b": {"decode_tps_aggregate": 20.0, "profile": {"kv": "fp8"}},
        },
    }


def test_interleaved_ab_strips_separator_and_reports_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _tool_module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for worktree in (baseline, candidate):
        (worktree / "tests").mkdir(parents=True)
        (worktree / "tests" / "e2e_full_benchmark.py").touch()
    seen_args: list[list[str]] = []

    def fake_run(worktree: Path, benchmark_args: list[str], _output: Path):
        seen_args.append(benchmark_args)
        return _payload(fingerprint={"models": ["model-a", "model-b"]})

    monkeypatch.setattr(tool, "_run", fake_run)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interleaved_e2e_ab.py",
            "--baseline-worktree",
            str(baseline),
            "--candidate-worktree",
            str(candidate),
            "--runs",
            "3",
            "--json-out",
            str(output),
            "--",
            "--models",
            "tinyllama",
        ],
    )

    tool.main()

    assert all(args == ["--models", "tinyllama"] for args in seen_args)
    result = json.loads(output.read_text())
    assert set(result["per_model"]) == {"model-a", "model-b"}


def test_interleaved_ab_writes_results_before_fingerprint_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _tool_module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for worktree in (baseline, candidate):
        (worktree / "tests").mkdir(parents=True)
        (worktree / "tests" / "e2e_full_benchmark.py").touch()
    calls = 0

    def fake_run(_worktree: Path, _benchmark_args: list[str], _output: Path):
        nonlocal calls
        calls += 1
        fingerprint = {"models": ["model-a"]}
        if calls == 2:
            fingerprint["gpu"] = "different"
        return _payload(fingerprint=fingerprint)

    monkeypatch.setattr(tool, "_run", fake_run)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interleaved_e2e_ab.py",
            "--baseline-worktree",
            str(baseline),
            "--candidate-worktree",
            str(candidate),
            "--json-out",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="raw results were written"):
        tool.main()

    assert output.is_file()
