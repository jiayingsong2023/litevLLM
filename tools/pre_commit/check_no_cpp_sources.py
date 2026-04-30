#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject C/C++/CUDA source files in the lite-only tree."""

from __future__ import annotations

import pathlib
import subprocess

FORBIDDEN_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}
ALLOWED_PREFIXES = (pathlib.Path("vllm/third_party"),)


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    violations: list[str] = []
    for raw in result.stdout.splitlines():
        path = pathlib.Path(raw)
        if any(path == prefix or prefix in path.parents for prefix in ALLOWED_PREFIXES):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.parts[:1] == ("csrc",):
            violations.append(raw)

    if violations:
        print("C/C++/CUDA sources are forbidden in FastInference lite-only code:")
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
