#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject direct Triton imports outside the central shim."""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPORT_RE = re.compile(r"^\s*(import\s+triton\b|from\s+triton\b)")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}
EXCLUDED_PREFIXES = (
    pathlib.Path("vllm/third_party"),
    pathlib.Path("vllm/triton_utils"),
)


def _is_excluded(path: pathlib.Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    return any(rel == prefix or prefix in rel.parents for prefix in EXCLUDED_PREFIXES)


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        if _is_excluded(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if IMPORT_RE.match(line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    if violations:
        print(
            "Direct Triton imports are forbidden; import via vllm.triton_utils instead."
        )
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
