# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.tools._gemma4_diag_utils import (
    parse_drift_metrics,
    parse_strict_metrics,
    warn_if_diff,
    warn_if_token_mismatch,
)

_ROOT = Path(__file__).resolve().parents[1]
_STRICT_BASELINE_PATH = (
    _ROOT / "tests" / "tools" / "fixtures" / "gemma4_a_strict_baseline.json"
)
_DRIFT_BASELINE_PATH = (
    _ROOT / "tests" / "tools" / "fixtures" / "gemma4_layer_drift_baseline_short_hi.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_default_gemma_model_path() -> str | None:
    module_path = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_single_prompt_smoke", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_default_model_path()


def test_parse_helpers_cover_sample_formats() -> None:
    strict_text = (
        "Prefill Logits -> CosSim: 0.999983, MaxErr: 0.000000\n"
        "Prefill Token: HF(argmax)=537 | Lite(engine)=537 | Lite(argmax logits)=537\n"
    )
    strict = parse_strict_metrics(strict_text)
    assert strict["cos_sim"] == pytest.approx(0.999983)
    assert strict["hf_argmax"] == 537

    drift_text = (
        "[Drift] token_to_step={1: 2, 16: 17, 24: 25, 32: 33}\n\n"
        "[token=16]\n"
        "  local: cos_to_t1 mean=0.726229 min=0.139810 cos_to_prev mean=0.726229 min=0.139810\n"
        "  full: cos_to_t1 mean=0.719264 min=0.065373 cos_to_prev mean=0.719264 min=0.065373\n"
    )
    drift = parse_drift_metrics(drift_text)
    assert drift["token_to_step"][16] == 17
    assert drift["metrics"][16]["local"]["cos_to_t1_mean"] == pytest.approx(0.726229)


def test_gemma4_diagnostics_warn_only_against_baseline() -> None:
    if os.environ.get("RUN_GEMMA4_DIAGNOSTIC", "0") != "1":
        pytest.skip(
            "Set RUN_GEMMA4_DIAGNOSTIC=1 to run Gemma4 diagnostics warn-only audit."
        )
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA/ROCm unavailable; skipping Gemma4 diagnostics warn-only audit."
        )

    model = _resolve_default_gemma_model_path()
    if not model or not Path(model).is_dir():
        pytest.skip(
            "Gemma4 local model dir not found; skipping diagnostics warn-only audit."
        )

    strict_base = _load_json(_STRICT_BASELINE_PATH)
    drift_base = _load_json(_DRIFT_BASELINE_PATH)

    env = os.environ.copy()
    env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = "1"
    env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = "512"

    strict_cmd = [
        "uv",
        "run",
        "python",
        "tests/tools/gemma4_prefill_strict_audit.py",
        "--model",
        model,
        "--hf-device",
        "cuda",
    ]
    strict_proc = subprocess.run(
        strict_cmd,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert strict_proc.returncode == 0, strict_proc.stdout + "\n" + strict_proc.stderr
    strict = parse_strict_metrics(strict_proc.stdout)

    warn_if_diff(
        "strict.cos_sim", strict["cos_sim"], float(strict_base["cos_sim"]), 0.001
    )
    warn_if_diff(
        "strict.max_err", strict["max_err"], float(strict_base["max_err"]), 0.05
    )
    warn_if_token_mismatch(
        "strict.hf_argmax", strict["hf_argmax"], int(strict_base["hf_argmax"])
    )
    warn_if_token_mismatch(
        "strict.lite_engine_argmax",
        strict["lite_engine_argmax"],
        int(strict_base["lite_engine_argmax"]),
    )
    warn_if_token_mismatch(
        "strict.lite_logits_argmax",
        strict["lite_logits_argmax"],
        int(strict_base["lite_logits_argmax"]),
    )

    drift_cmd = [
        "uv",
        "run",
        "python",
        "tests/tools/gemma4_layer_drift_diagnostic.py",
        "--model",
        model,
        "--prompt-id",
        str(drift_base["prompt_id"]),
        "--checkpoints",
        ",".join(str(int(x)) for x in drift_base["checkpoints"][1:]),
        "--max-new-tokens",
        str(int(drift_base["max_new_tokens"])),
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "1024",
    ]
    drift_proc = subprocess.run(
        drift_cmd,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert drift_proc.returncode == 0, drift_proc.stdout + "\n" + drift_proc.stderr
    drift = parse_drift_metrics(drift_proc.stdout)

    tol = drift_base["warn_tolerance"]
    step_tol = int(tol["token_to_step_abs"])
    for key_str, exp_step in drift_base["token_to_step"].items():
        token = int(key_str)
        got_step = int(drift["token_to_step"].get(token, -999999))
        if abs(got_step - int(exp_step)) > step_tol:
            warnings.warn(
                f"[Gemma4DiagWarn] drift.token_to_step[{token}] changed: "
                f"actual={got_step} expected={exp_step} tol={step_tol}",
                stacklevel=2,
            )

    mean_tol = float(tol["mean_abs_diff"])
    min_tol = float(tol["min_abs_diff"])
    for cp_str, cp_metrics in drift_base["metrics"].items():
        cp = int(cp_str)
        for kind in ("local", "full"):
            got = drift["metrics"].get(cp, {}).get(kind)
            if got is None:
                warnings.warn(
                    f"[Gemma4DiagWarn] drift missing metrics for token={cp}, kind={kind}",
                    stacklevel=2,
                )
                continue
            exp = cp_metrics[kind]
            warn_if_diff(
                f"drift[{cp}].{kind}.cos_to_t1_mean",
                float(got["cos_to_t1_mean"]),
                float(exp["cos_to_t1_mean"]),
                mean_tol,
            )
            warn_if_diff(
                f"drift[{cp}].{kind}.cos_to_prev_mean",
                float(got["cos_to_prev_mean"]),
                float(exp["cos_to_prev_mean"]),
                mean_tol,
            )
            warn_if_diff(
                f"drift[{cp}].{kind}.cos_to_t1_min",
                float(got["cos_to_t1_min"]),
                float(exp["cos_to_t1_min"]),
                min_tol,
            )
            warn_if_diff(
                f"drift[{cp}].{kind}.cos_to_prev_min",
                float(got["cos_to_prev_min"]),
                float(exp["cos_to_prev_min"]),
                min_tol,
            )
