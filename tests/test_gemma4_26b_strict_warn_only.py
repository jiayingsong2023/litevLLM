# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.tools._gemma4_diag_utils import (
    parse_strict_metrics,
    warn_if_diff,
    warn_if_token_mismatch,
)

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_PATH = _ROOT / "tests" / "tools" / "fixtures" / "gemma4_26b_a_strict_baseline.json"

_MODEL_CANDIDATES: tuple[str, ...] = (
    "models/gemma-4-26B-A4B-it-AWQ-4bit",
    "models/Gemma-4-26B-A4B-it-AWQ-4bit",
    "models/gemma-4-26b-a4b-it-awq-4bit",
    "models/Gemma-4-26B-A4B",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_path() -> str | None:
    env_model = os.environ.get("MODEL_GEMMA4_26B_A4B", "").strip()
    if env_model:
        return env_model
    for candidate in _MODEL_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


def test_gemma4_26b_strict_baseline_fixture_schema() -> None:
    base = _load_json(_BASELINE_PATH)
    for key in (
        "model",
        "quant",
        "preset",
        "prompt_id",
        "hf_device",
        "kv_type",
        "max_model_len",
        "max_num_batched_tokens",
        "cos_sim",
        "max_err",
        "hf_argmax",
        "lite_engine_argmax",
        "lite_logits_argmax",
    ):
        assert key in base, f"missing key: {key}"


def test_gemma4_26b_strict_warn_only_against_baseline() -> None:
    if os.environ.get("RUN_GEMMA4_26B_DIAGNOSTIC", "0") != "1":
        pytest.skip("Set RUN_GEMMA4_26B_DIAGNOSTIC=1 to run Gemma4-26B strict warn-only audit.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm unavailable; skipping Gemma4-26B strict warn-only audit.")

    model = _resolve_model_path()
    if not model or not Path(model).is_dir():
        pytest.skip("Gemma4-26B local model dir not found; skipping strict warn-only audit.")

    base = _load_json(_BASELINE_PATH)
    env = os.environ.copy()
    env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = "1"
    env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(int(base["max_model_len"]))

    cmd = [
        "uv",
        "run",
        "python",
        "tests/tools/gemma4_prefill_strict_audit.py",
        "--model",
        model,
        "--hf-model",
        model,
        "--preset",
        "gemma4_26b_a4b",
        "--hf-device",
        "cuda",
        "--kv-type",
        "turbo_int4",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    got = parse_strict_metrics(proc.stdout)

    warn_if_diff("strict.cos_sim", got["cos_sim"], float(base["cos_sim"]), 0.001, tag="Gemma4-26BDiagWarn")
    warn_if_diff("strict.max_err", got["max_err"], float(base["max_err"]), 0.05, tag="Gemma4-26BDiagWarn")
    warn_if_token_mismatch("strict.hf_argmax", got["hf_argmax"], int(base["hf_argmax"]), tag="Gemma4-26BDiagWarn")
    warn_if_token_mismatch(
        "strict.lite_engine_argmax",
        got["lite_engine_argmax"],
        int(base["lite_engine_argmax"]),
        tag="Gemma4-26BDiagWarn",
    )
    warn_if_token_mismatch(
        "strict.lite_logits_argmax",
        got["lite_logits_argmax"],
        int(base["lite_logits_argmax"]),
        tag="Gemma4-26BDiagWarn",
    )
