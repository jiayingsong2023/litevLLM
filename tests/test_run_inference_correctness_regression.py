# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_run_inference_correctness_regression_wraps_stages_with_timeout(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    timeout_log = tmp_path / "timeout_calls.log"
    fake_timeout = fake_bin / "timeout"
    fake_timeout.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'printf "%s\\n" "$*" >> "{timeout_log}"',
                'while [[ "$1" == --* ]]; do shift; done',
                "shift",
                '"$@"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_timeout.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    qwen_dir = tmp_path / "models" / "Qwen3.5-9B-AWQ"
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "0"
    env["SKIP_A_TIER"] = "1"
    env["FI_CORRECTNESS_STAGE_TIMEOUT"] = "12s"
    env["FI_CORRECTNESS_STAGE_KILL_AFTER"] = "3s"
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
    calls = timeout_log.read_text(encoding="utf-8")
    assert "--kill-after=3s 12s env FASTINFERENCE_CONFIG=" in calls
    assert "uv run python tests/tools/quality_bar_spotcheck.py" in calls
    assert "[Stage] START Tier-B TinyLlama spotcheck timeout=12s" in proc.stdout
    assert "[Stage] OK Tier-B Qwen3.5-9B AWQ spotcheck" in proc.stdout


def test_run_inference_correctness_regression_skips_gemma4_31b_a_strict(
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
    env["RUN_GEMMA4_26B"] = "0"
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
    assert "run python tests/tools/gemma4_prefill_strict_audit.py" not in calls
    assert str(gemma_dir) in calls
    assert "run python tests/tools/gemma4_single_prompt_smoke.py" not in calls


def test_run_inference_correctness_regression_runs_opt_in_deepseek_gpu_smoke(
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
    deepseek_path = tmp_path / "models" / "deepseek-v4-flash.gguf"
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    deepseek_path.parent.mkdir(parents=True, exist_ok=True)
    deepseek_path.write_bytes(b"gguf")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["MODEL_DEEPSEEK_V4_FLASH_GGUF"] = str(deepseek_path)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "0"
    env["RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE"] = "1"
    env["SKIP_A_TIER"] = "1"
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
    calls = log_path.read_text(encoding="utf-8")
    assert "run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py" in calls
    assert f"--model {deepseek_path}" in calls
    assert "DeepSeek V4 Flash GPU smoke" in proc.stdout


def test_run_inference_correctness_regression_perf_diag_includes_deepseek(
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
    deepseek_path = tmp_path / "models" / "deepseek-v4-flash.gguf"
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    deepseek_path.parent.mkdir(parents=True, exist_ok=True)
    deepseek_path.write_bytes(b"gguf")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["MODEL_DEEPSEEK_V4_FLASH_GGUF"] = str(deepseek_path)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "0"
    env["RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE"] = "1"
    env["RUN_PERF_DIAG"] = "1"
    env["SKIP_A_TIER"] = "1"
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
    calls = log_path.read_text(encoding="utf-8")
    assert "run python tests/e2e_full_benchmark.py" in calls
    assert "--models tinyllama,qwen35_9b_awq,deepseek_v4_flash_q2_gguf" in calls


def test_run_inference_correctness_regression_runs_large_gemma_a_tier_by_default(
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
                (
                    'printf "CONFIG=%s KV_LEN=%s ACTIVE=%s CMD=%s\n" '
                    '"${FASTINFERENCE_CONFIG:-}" '
                    '"${FASTINFERENCE_KV_MAX_MODEL_LEN:-}" '
                    '"${FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS:-}" '
                    '"$*" >> "' + str(log_path) + '"'
                ),
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    qwen_awq_dir = tmp_path / "models" / "Qwen3.5-9B-AWQ"
    qwen_fp16_dir = tmp_path / "models" / "Qwen3.5-9B-FP16"
    gemma31_dir = tmp_path / "models" / "gemma-4-31B-it-AWQ-4bit"
    gemma26_dir = tmp_path / "models" / "gemma-4-26B-A4B-it-AWQ-4bit"
    for model_dir in (tiny_dir, qwen_awq_dir, qwen_fp16_dir, gemma31_dir, gemma26_dir):
        model_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_awq_dir)
    env["HF_QWEN35_9B_FP16"] = str(qwen_fp16_dir)
    env["MODEL_GEMMA4_31B_Q4"] = str(gemma31_dir)
    env["MODEL_GEMMA4_26B_A4B"] = str(gemma26_dir)
    env["RUN_GEMMA4_31B"] = "1"
    env["RUN_GEMMA4_26B"] = "1"
    env.pop("RUN_GEMMA4_LARGE_A_TIER", None)
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
    calls = log_path.read_text(encoding="utf-8")
    assert "run python tests/tools/quality_bar_spotcheck.py" in calls
    assert str(gemma31_dir) in calls
    assert str(gemma26_dir) in calls
    assert "run python tests/tools/gemma4_prefill_strict_audit.py" in calls
    assert "--preset gemma4_26b_a4b" in calls
    assert "run python tests/tools/gemma4_single_prompt_smoke.py" in calls
    assert f"--model {gemma31_dir}" in calls
    assert f"--model {gemma26_dir}" in calls
    assert (
        "KV_LEN=512 ACTIVE=1 CMD=run python tests/tools/gemma4_single_prompt_smoke.py"
        in calls
    )
    assert "Gemma4-31B A-strict prefill audit is disabled" in proc.stdout
    assert "Skipping Gemma4 large-model A-tier" not in proc.stdout
    assert "Cleanup after Gemma4-31B A-lite" in proc.stdout
    assert "Cleanup after Gemma4-26B A-lite" in proc.stdout


def test_run_inference_correctness_regression_requires_qwen35_fp16_reference(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    qwen_awq_dir = tmp_path / "models" / "Qwen3.5-9B-AWQ"
    missing_qwen_fp16_dir = tmp_path / "models" / "Qwen3.5-9B-FP16"
    tiny_dir.mkdir(parents=True)
    qwen_awq_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_awq_dir)
    env["HF_QWEN35_9B_FP16"] = str(missing_qwen_fp16_dir)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "0"
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")

    proc = subprocess.run(
        ["bash", "tests/run_inference_correctness_regression.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Missing model directory for Qwen3.5-9B-FP16" in proc.stdout


def test_run_inference_correctness_regression_uses_config_for_gemma4_26b(
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
                (
                    'printf "CONFIG=%s CMD=%s\\n" '
                    '"${FASTINFERENCE_CONFIG:-}" '
                    '"$*" >> "' + str(log_path) + '"'
                ),
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    qwen_dir = tmp_path / "models" / "Qwen3.5-9B-AWQ"
    gemma26_dir = tmp_path / "models" / "gemma-4-26B-A4B-it-AWQ-4bit"
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    gemma26_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["MODEL_GEMMA4_26B_A4B"] = str(gemma26_dir)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "1"
    env["RUN_GEMMA4_26B_A_LITE"] = "1"
    env["SKIP_A_TIER"] = "1"
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
    assert "run python tests/tools/quality_bar_spotcheck.py" in calls
    assert str(gemma26_dir) in calls
    assert "CONFIG=/tmp/fastinference-correctness-config." in calls
    assert "gemma26b-benchmark-turbo.toml" in calls


def test_run_inference_correctness_regression_uses_default_local_gemma_paths(
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
                f'printf "%s\\n" "$*" >> "{log_path}"',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    tiny_rel = Path("models/TinyLlama-1.1B-Chat-v1.0")
    qwen_rel = Path("models/Qwen3.5-9B-AWQ")
    gemma31_rel = Path("models/gemma-4-31B-it-AWQ-4bit")
    gemma26_rel = Path("models/gemma-4-26B-A4B-it-AWQ-4bit")
    (repo_root / tiny_rel).mkdir(parents=True, exist_ok=True)
    (repo_root / qwen_rel).mkdir(parents=True, exist_ok=True)
    (repo_root / gemma31_rel).mkdir(parents=True, exist_ok=True)
    (repo_root / gemma26_rel).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env.pop("MODEL_GEMMA4_31B_Q4", None)
    env.pop("MODEL_GEMMA4_26B_A4B", None)
    env["SKIP_A_TIER"] = "1"
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
    calls = log_path.read_text(encoding="utf-8")
    assert str(gemma31_rel) in calls
    assert str(gemma26_rel) in calls


def test_run_inference_correctness_regression_uses_tinyllama_prompts_file(
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
    tiny_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["MODEL_TINYLLAMA"] = str(tiny_dir)
    env["MODEL_QWEN35_9B_AWQ"] = str(qwen_dir)
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "0"
    env["SKIP_A_TIER"] = "1"
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
    calls = log_path.read_text(encoding="utf-8")
    assert (
        "--prompts-file tests/tools/fixtures/tinyllama_correctness_prompts_default.json"
        in calls
    )
