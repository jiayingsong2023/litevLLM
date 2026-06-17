# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_IQ2_XXS,
)


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


def _load_shape_tool() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "inspect_deepseek_v4_flash_expert_shapes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "inspect_deepseek_v4_flash_expert_shapes",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_quality_tool() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "deepseek_v4_flash_quality_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_flash_quality_smoke",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_tool()
shape_tool = _load_shape_tool()
quality = _load_quality_tool()


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


def test_quality_parser_accepts_readability_args(tmp_path: Path) -> None:
    json_path = tmp_path / "quality.json"

    args = quality.parse_args(
        [
            "--model",
            "model.gguf",
            "--prompt-text",
            "What is the capital of France?",
            "--max-tokens",
            "8",
            "--min-output-chars",
            "12",
            "--json-out",
            str(json_path),
        ]
    )

    assert args.model == Path("model.gguf")
    assert args.prompt_text == "What is the capital of France?"
    assert args.max_tokens == 8
    assert args.min_output_chars == 12
    assert args.json_out == json_path


def test_quality_readability_rejects_repeated_token_garbage() -> None:
    verdict = quality.evaluate_readability(
        text="aaaaaa",
        generated_token_ids=[7, 7, 7, 7, 7, 7],
        min_output_chars=3,
    )

    assert verdict["passed"] is False
    assert "token_repeat" in verdict["reasons"]


def test_quality_readability_rejects_fragmented_word_repetition() -> None:
    verdict = quality.evaluate_readability(
        text="/gen/genazzo/genazzo/gen(The disposed",
        generated_token_ids=[80566, 80566, 62594, 80566, 62594, 80566, 96093],
        min_output_chars=8,
    )

    assert verdict["passed"] is False
    assert "word_repeat" in verdict["reasons"]


def test_quality_readability_accepts_short_coherent_text() -> None:
    verdict = quality.evaluate_readability(
        text="Paris is the capital of France.",
        generated_token_ids=[100, 42, 51, 79, 12, 9],
        min_output_chars=8,
    )

    assert verdict["passed"] is True
    assert verdict["reasons"] == []


def test_quality_decodes_token_ids_with_gguf_tokens() -> None:
    text = quality.decode_generated_tokens(
        [0, 1, 2],
        gguf_tokens=["<s>", "Paris", " is"],
    )

    assert text == "Paris is"


def test_quality_simple_prompt_encoder_uses_longest_gguf_pieces() -> None:
    token_ids = quality.encode_prompt_text(
        "What is?",
        gguf_tokens=["<s>", "What", " is", "?", "What is"],
    )

    assert token_ids == [4, 3]


def test_shape_inspector_parser_accepts_limit_and_json_args(tmp_path: Path) -> None:
    json_path = tmp_path / "shapes.json"

    args = shape_tool.parse_args(
        [
            "--model",
            "model.gguf",
            "--limit",
            "12",
            "--json",
            str(json_path),
        ]
    )

    assert args.model == Path("model.gguf")
    assert args.limit == 12
    assert args.json == json_path


def test_shape_inspector_schema_uses_grouped_expert_dims() -> None:
    tensor = shape_tool.DeepSeekV4FlashTensor(
        name="blk.2.ffn_gate_exps.weight",
        dims=(2048, 512, 256),
        tensor_type=GGML_TYPE_IQ2_XXS,
        offset=0,
        nbytes=0,
    )

    record = shape_tool.expert_shape_record(
        layer_idx=2,
        projection="gate",
        tensor=tensor,
    )

    assert record == {
        "layer_idx": 2,
        "projection": "gate",
        "tensor_name": "blk.2.ffn_gate_exps.weight",
        "ggml_type": GGML_TYPE_IQ2_XXS,
        "rows": 512,
        "columns": 2048,
        "expert_count": 256,
        "columns_blocks": 8,
        "nbytes_per_expert": 2048 * 2 * 66,
    }


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "profile.json"

    smoke.write_json(output_path, {"ok": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def test_phase3_metrics_schema_is_stable() -> None:
    metrics = smoke.phase3_metrics(
        profile={"counters": {"deepseek_prefetch_failures": 1}},
        gpu_staging={
            "lru_evictions": 2,
            "streamed_bytes": 3,
            "prefetch_hits": 4,
            "prefetch_misses": 5,
            "prefetch_failures": 6,
        },
        gpu_backend={"quantized_expert_calls": 7},
    )

    assert metrics == {
        "cpu_sync_points": 0,
        "lru_evictions": 2,
        "prefetch_failures": 7,
        "prefetch_hits": 4,
        "prefetch_misses": 5,
        "quantized_expert_calls": 7,
        "streamed_bytes": 3,
    }


def test_phase4_metrics_schema_is_stable() -> None:
    metrics = smoke.phase4_metrics(
        profile={},
        gpu_staging={},
        gpu_backend={
            "q2_k_triton_calls": 2,
            "iq2_xxs_triton_calls": 3,
            "iq2_xxs_gate_up_fused_calls": 5,
            "q2_iq2_reference_fallback_calls": 4,
        },
    )

    assert metrics == {
        "iq2_xxs_gate_up_fused_calls": 5,
        "iq2_xxs_triton_calls": 3,
        "q2_iq2_reference_fallback_calls": 4,
        "q2_k_triton_calls": 2,
    }


def test_phase4_metrics_defaults_missing_backend_counters_to_zero() -> None:
    metrics = smoke.phase4_metrics(profile={}, gpu_staging={}, gpu_backend={})

    assert metrics == {
        "iq2_xxs_gate_up_fused_calls": 0,
        "iq2_xxs_triton_calls": 0,
        "q2_iq2_reference_fallback_calls": 0,
        "q2_k_triton_calls": 0,
    }


def test_usable_inference_metrics_schema_is_stable() -> None:
    metrics = smoke.usable_inference_metrics(
        profile={"counters": {"deepseek_prefetch_failures": 2}},
        gpu_staging={
            "lru_evictions": 3,
            "pinned_entries": 4,
            "prefetch_failures": 5,
            "prefetch_hits": 6,
            "prefetch_misses": 7,
            "routed_expert_id_materializations": 8,
            "streamed_bytes": 9,
        },
        gpu_backend={
            "iq2_xxs_gate_up_fused_calls": 10,
            "q2_iq2_reference_fallback_calls": 11,
        },
    )

    assert metrics == {
        "iq2_xxs_gate_up_fused_calls": 10,
        "lru_evictions": 3,
        "pinned_entries": 4,
        "prefetch_failures": 7,
        "prefetch_hits": 6,
        "prefetch_misses": 7,
        "q2_iq2_reference_fallback_calls": 11,
        "routed_expert_id_materializations": 8,
        "streamed_bytes": 9,
    }
