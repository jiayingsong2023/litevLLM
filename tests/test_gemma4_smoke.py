# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm.model_executor.models.gemma4 import _assert_text_only_kwargs

_ROOT = Path(__file__).resolve().parents[1]


def _load_gemma4_smoke_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location("gemma4_single_prompt_smoke", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gemma4_smoke_mod() -> Any:
    return _load_gemma4_smoke_module()


@pytest.fixture(scope="module")
def gemma4_model_path(gemma4_smoke_mod: Any) -> str:
    path = gemma4_smoke_mod.resolve_default_model_path()
    if not path:
        pytest.skip("Gemma4 local model dir not found; skipping Gemma4 smoke tests.")
    if not Path(path).is_dir():
        pytest.skip(f"Gemma4 model path is not a local directory: {path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable; skipping Gemma4 smoke tests.")
    return path


@pytest.fixture(scope="module")
def gemma4_engine(gemma4_smoke_mod: Any, gemma4_model_path: str) -> Any:
    args = gemma4_smoke_mod._build_parser().parse_args(
        [
            "--model",
            gemma4_model_path,
            "--max-new-tokens",
            "8",
            "--temperature",
            "0",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            "512",
            "--min-output-chars",
            "1",
            "--max-num-batched-tokens",
            "1024",
        ]
    )
    engine, tokenizer, _ = gemma4_smoke_mod._build_engine(args)
    return {
        "args": args,
        "engine": engine,
        "tokenizer": tokenizer,
    }


def test_resolve_default_model_path_prefers_local_candidates(
    gemma4_smoke_mod: Any,
) -> None:
    path = gemma4_smoke_mod.resolve_default_model_path()
    if path is None:
        pytest.skip("Gemma4 local model dir not found.")
    assert Path(path).exists()


def test_build_engine_applies_max_model_len_to_model_config(
    gemma4_smoke_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeEngine:
        def __init__(self, vllm_config: Any) -> None:
            captured["vllm_config"] = vllm_config

    monkeypatch.setattr(gemma4_smoke_mod, "LiteEngine", FakeEngine)
    monkeypatch.setattr(
        gemma4_smoke_mod, "get_tokenizer", lambda *args, **kwargs: object()
    )

    args = gemma4_smoke_mod._build_parser().parse_args(
        [
            "--model",
            "dummy-gemma4",
            "--max-model-len",
            "512",
            "--max-num-batched-tokens",
            "1024",
        ]
    )

    gemma4_smoke_mod._build_engine(args)

    cfg = captured["vllm_config"]
    assert cfg.model_config.max_model_len == 512
    assert cfg.scheduler_config.max_model_len == 512
    assert cfg.scheduler_config.max_num_seqs == 1


def test_gemma4_single_text_generation_finishes(
    gemma4_smoke_mod: Any, gemma4_engine: dict[str, Any]
) -> None:
    result = gemma4_smoke_mod._run_single_request(
        engine=gemma4_engine["engine"],
        tokenizer=gemma4_engine["tokenizer"],
        prompt_id="pytest_smoke",
        prompt="Say hello in one short sentence.",
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        min_output_chars=1,
    )
    text, first_token_id, steps, issues = result
    assert first_token_id >= 0
    assert steps > 0
    assert text.strip()
    assert issues == []


def test_text_only_kwargs_reject_multimodal_inputs() -> None:
    with pytest.raises(
        NotImplementedError, match="text-only path does not support multimodal input"
    ):
        _assert_text_only_kwargs({"pixel_values": torch.zeros(1)})
