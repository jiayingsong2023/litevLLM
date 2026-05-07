# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_smoke_workflow_pytest_target_exists() -> None:
    workflow = _read(".github/workflows/smoke.yml")
    targets = re.findall(r"uv run pytest(?: -q)? ([^\n]+)", workflow)

    assert targets, "smoke workflow must run pytest against an explicit target"
    for target in targets:
        path = ROOT / target.strip()
        assert path.exists(), f"smoke workflow target is missing: {target}"


def test_stability_summary_smoke_files_exist() -> None:
    summary = _read("docs/STABILITY_WORK_SUMMARY.md")
    smoke_files = [
        path
        for path in re.findall(r"`(tests/smoke/[^`]+\.py)`", summary)
        if "*" not in path
    ]

    assert smoke_files, "stability summary must list smoke test files"
    missing = [path for path in smoke_files if not (ROOT / path).is_file()]
    assert not missing, "documented smoke files are missing: " + ", ".join(missing)


def test_capability_matrix_is_documented_and_referenced() -> None:
    matrix_path = ROOT / "docs/CAPABILITY_MATRIX.md"
    assert matrix_path.is_file(), "docs/CAPABILITY_MATRIX.md must exist"

    matrix = matrix_path.read_text(encoding="utf-8")
    for required in ("Supported", "Experimental", "Compatibility", "Unsupported"):
        assert required in matrix

    referenced_docs = (
        "README.md",
        "docs/models/supported_models.md",
        "docs/LITE_ONLY_STATUS.md",
    )
    for doc in referenced_docs:
        text = _read(doc)
        assert "CAPABILITY_MATRIX.md" in text, f"{doc} must reference capability matrix"


def test_runtime_factory_does_not_read_fastinference_env_directly() -> None:
    factory = _read("vllm/engine/runtime_factory.py")

    forbidden_patterns = (
        "os.environ",
        "getenv",
        "FASTINFERENCE_",
    )
    for pattern in forbidden_patterns:
        assert pattern not in factory, (
            "runtime_factory must receive runtime policy from RuntimeConfig, "
            f"not read env directly via {pattern!r}"
        )
