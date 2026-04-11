# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_run_inference_correctness_regression_invokes_gemma4_a_strict(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    log_path = tmp_path / "uv_calls.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'printf "%s\\n" "$*" >> "{log_path}"',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    qwen_dir = tmp_path / "models" / "Qwen3.5-9B-AWQ"
    gemma_dir = tmp_path / "models" / "gemma-4-31B-it-AWQ-4bit"
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    gemma_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["MODEL_GEMMA4_31B_Q4"] = str(gemma_dir)
    env["RUN_GEMMA4_31B"] = "1"
    env["RUN_GEMMA4_A_STRICT"] = "1"
    env["RUN_GEMMA4_A_LITE"] = "0"
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")

    proc = subprocess.run(
        ["bash", "tests/run_inference_correctness_regression.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert log_path.exists()

    calls = log_path.read_text(encoding="utf-8")
    assert "run python tests/tools/gemma4_prefill_strict_audit.py" in calls
    assert "--model" in calls
    assert str(gemma_dir) in calls
    assert "--hf-device cuda" in calls
    assert "run python tests/tools/gemma4_single_prompt_smoke.py" not in calls
