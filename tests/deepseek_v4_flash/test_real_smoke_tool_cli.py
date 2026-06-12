# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_smoke_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "run_deepseek_v4_flash_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "run_deepseek_v4_flash_gpu_smoke",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_tool()


def test_smoke_parser_accepts_profile_and_repeat_args(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles" / "smoke.json"

    args = smoke.parse_args(
        [
            "--model",
            "model.gguf",
            "--context-length",
            "4096",
            "--max-tokens",
            "8",
            "--repeat",
            "2",
            "--profile-json",
            str(profile_path),
        ]
    )

    assert args.model == Path("model.gguf")
    assert args.context_length == 4096
    assert args.max_tokens == 8
    assert args.repeat == 2
    assert args.profile_json == profile_path


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "profile.json"

    smoke.write_json(output_path, {"ok": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}
