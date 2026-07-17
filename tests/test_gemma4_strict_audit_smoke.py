# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_gemma4_strict_audit_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_prefill_strict_audit.py"
    spec = importlib.util.spec_from_file_location("gemma4_prefill_strict_audit", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gemma4_strict_audit_mod() -> Any:
    return _load_gemma4_strict_audit_module()


def test_build_parser_uses_discovered_default_model(
    gemma4_strict_audit_mod: Any, tmp_path: Path
) -> None:
    model_dir = tmp_path / "gemma4-model"
    model_dir.mkdir()
    gemma4_strict_audit_mod._default_model_path = lambda: str(model_dir)

    args = gemma4_strict_audit_mod._build_parser().parse_args([])

    assert args.model == str(model_dir)
    assert args.hf_model is None
    assert args.hf_device == "cuda"
    assert args.prompt_id == "en_capital"
    assert args.quant == "awq"


def test_main_rejects_non_local_model_dir(
    gemma4_strict_audit_mod: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["gemma4_prefill_strict_audit.py", "--model", "missing-model"]
    )

    rc = gemma4_strict_audit_mod.main()

    captured = capsys.readouterr()
    assert rc == 2
    assert "[A-strict][ERROR] local model dir required" in captured.out


def test_main_builds_verify_command_and_env(
    gemma4_strict_audit_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "gemma4-awq"
    hf_dir = tmp_path / "gemma4-fp16-ref"
    model_dir.mkdir()
    hf_dir.mkdir()

    calls: dict[str, Any] = {}

    def _fake_run(cmd: list[str], cwd: str, env: dict[str, str], check: bool) -> Any:
        calls["cmd"] = list(cmd)
        calls["cwd"] = cwd
        calls["env"] = dict(env)
        calls["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(gemma4_strict_audit_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_prefill_strict_audit.py",
            "--model",
            str(model_dir),
            "--hf-model",
            str(hf_dir),
            "--prompt-id",
            "zh_capital",
            "--max-model-len",
            "320",
            "--max-num-batched-tokens",
            "640",
            "--gpu-memory-utilization",
            "0.75",
            "--kv-type",
            "turbo_int4",
            "--quant",
            "awq",
        ],
    )

    rc = gemma4_strict_audit_mod.main()

    assert rc == 7
    assert calls["cwd"] == str(_ROOT)
    assert calls["check"] is False
    assert calls["cmd"] == [
        sys.executable,
        "tests/verify_semantic_integrity.py",
        "--model",
        str(model_dir),
        "--preset",
        "gemma4_31b_q4",
        "--quant",
        "awq",
        "--prompt",
        gemma4_strict_audit_mod.DEFAULT_PROMPTS["zh_capital"],
        "--hf-model",
        str(hf_dir),
        "--hf-device",
        "cuda",
        "--max-model-len",
        "320",
        "--max-num-batched-tokens",
        "640",
        "--gpu-memory-utilization",
        "0.75",
        "--prefill-only",
        "--apply-chat-template",
        "off",
        "--report-first-drift-layer",
        "--drift-cos-threshold",
        "0.995",
    ]
    assert calls["env"]["FASTINFERENCE_KV_TYPE"] == "turbo_int4"
    assert calls["env"]["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] == "1"
    assert calls["env"]["FASTINFERENCE_KV_MAX_MODEL_LEN"] == "320"
