# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_run_gemma4_26b_diagnostics_warn_only_sets_env_and_invokes_pytest(
    tmp_path: Path,
) -> None:
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
                f'printf "RUN_GEMMA4_26B_DIAGNOSTIC=%s CMD=%s\\n" "${{RUN_GEMMA4_26B_DIAGNOSTIC:-}}" "$*" >> "{log_path}"',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")

    proc = subprocess.run(
        ["bash", "tests/run_gemma4_26b_diagnostics_warn_only.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert "RUN_GEMMA4_26B_DIAGNOSTIC=1" in calls
    assert "pytest tests/test_gemma4_26b_strict_warn_only.py -q" in calls
