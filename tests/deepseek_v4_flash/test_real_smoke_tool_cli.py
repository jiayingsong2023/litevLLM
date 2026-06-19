# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest
from pytest import MonkeyPatch

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
    path = Path(__file__).parents[1] / "tools" / "deepseek_v4_flash_quality_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_flash_quality_smoke",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_e2e_benchmark() -> ModuleType:
    path = Path(__file__).parents[1] / "e2e_full_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "e2e_full_benchmark",
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
e2e_benchmark = _load_e2e_benchmark()


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
            "--warmup-tokens",
            "4",
            "--repeat",
            "2",
            "--min-steady-decode-tps",
            "1.25",
            "--profile-json",
            str(profile_path),
        ]
    )

    assert args.model == Path("model.gguf")
    assert args.context_length == 4096
    assert args.max_tokens == 8
    assert args.warmup_tokens == 4
    assert args.repeat == 2
    assert args.min_steady_decode_tps == 1.25
    assert args.profile_json == profile_path


def test_smoke_ready_backend_enables_required_kernels() -> None:
    backend = smoke._build_ready_backend()

    assert backend.is_ready is True
    assert backend.missing_kernels == ()


def test_smoke_decode_metrics_from_token_times_uses_steady_state() -> None:
    metrics = smoke.decode_metrics_from_token_times([1200.0] + [900.0] * 15)

    assert metrics["decode_tokens_total"] == 16
    assert metrics["decode_ms_total"] == 14700.0
    assert metrics["decode_tps_agg"] == 16 * 1000.0 / 14700.0
    assert metrics["decode_tps_steady_state"] == 15 * 1000.0 / 13500.0


def test_smoke_steady_decode_tps_gate_is_disabled_by_default() -> None:
    smoke.validate_steady_decode_tps(
        decode_tps_steady_state=0.1,
        min_steady_decode_tps=0.0,
    )


def test_smoke_steady_decode_tps_gate_rejects_low_tps() -> None:
    with pytest.raises(SystemExit, match="0.90 < 1.00"):
        smoke.validate_steady_decode_tps(
            decode_tps_steady_state=0.9,
            min_steady_decode_tps=1.0,
        )


def test_smoke_steady_decode_tps_gate_accepts_threshold() -> None:
    smoke.validate_steady_decode_tps(
        decode_tps_steady_state=1.25,
        min_steady_decode_tps=1.0,
    )


def test_smoke_timed_generate_prefers_model_timed_api() -> None:
    class TimedModel:
        def generate_greedy_kernel_timed(
            self,
            input_ids: list[int],
            *,
            max_tokens: int,
        ) -> tuple[list[int], list[float]]:
            assert input_ids == [1]
            assert max_tokens == 2
            return [1, 2, 3], [12, 34]

    output_ids, token_elapsed_ms = smoke.generate_greedy_with_token_timings(
        TimedModel(),
        [1],
        max_tokens=2,
    )

    assert output_ids == [1, 2, 3]
    assert token_elapsed_ms == [12.0, 34.0]


def test_smoke_timed_generate_falls_back_to_single_token_calls(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeEvent:
        def record(self) -> None:
            pass

        def elapsed_time(self, _other: FakeEvent) -> float:
            return 10.0

    class FallbackModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[int], int]] = []

        def generate_greedy_kernel(
            self,
            input_ids: list[int],
            *,
            max_tokens: int,
        ) -> list[int]:
            self.calls.append((list(input_ids), max_tokens))
            return [*input_ids, len(input_ids) + 1]

    model = FallbackModel()
    monkeypatch.setattr(smoke.torch.cuda, "Event", lambda **_kwargs: FakeEvent())
    monkeypatch.setattr(smoke.torch.cuda, "synchronize", lambda: None)

    output_ids, token_elapsed_ms = smoke.generate_greedy_with_token_timings(
        model,
        [1],
        max_tokens=2,
    )

    assert model.calls == [([1], 1), ([1, 2], 1)]
    assert output_ids == [1, 2, 3]
    assert token_elapsed_ms == [10.0, 10.0]


def test_e2e_deepseek_parser_prefers_steady_state_decode_metrics() -> None:
    spec = e2e_benchmark.ModelSpec(
        key="deepseek",
        model_path="model.gguf",
        display_name="DeepSeek",
        quant="deepseek-v4-flash-gguf",
        concurrent_reqs=1,
        prompt_tokens_target=1,
        max_new_tokens=16,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_run_seconds=60,
        stable_env={},
    )
    payload = {
        "context_length": 4096,
        "max_tokens": 16,
        "repeat": 1,
        "runs": [
            {
                "elapsed_ms": 16000.0,
                "tokens_per_second": 1.0,
                "generated_token_count": 16,
                "token_elapsed_ms": [1200.0] + [900.0] * 15,
                "decode_tokens_total": 16,
                "decode_ms_total": 14700.0,
                "decode_tps_agg": 16 * 1000.0 / 14700.0,
                "decode_tps_steady_state": 15 * 1000.0 / 13500.0,
            }
        ],
    }

    result = e2e_benchmark._deepseek_smoke_payload_to_benchmark_result(
        spec,
        payload,
        wall_sec=20.0,
    )

    assert result["decode_tokens_total"] == 16
    assert result["decode_ms_total"] == 14700.0
    assert result["decode_tps_aggregate"] == 16 * 1000.0 / 14700.0
    assert result["decode_tps_p50"] == 15 * 1000.0 / 13500.0


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


def test_quality_parser_accepts_performance_gate_args(tmp_path: Path) -> None:
    args = quality.parse_args(
        [
            "--model",
            "model.gguf",
            "--warmup-tokens",
            "1",
            "--repeat",
            "3",
            "--min-decode-tps",
            "2.5",
            "--max-total-elapsed-ms",
            "1000.0",
        ]
    )

    assert args.model == Path("model.gguf")
    assert args.warmup_tokens == 1
    assert args.repeat == 3
    assert args.min_decode_tps == 2.5
    assert args.max_total_elapsed_ms == 1000.0


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


def test_quality_bpe_prompt_encoder_uses_gguf_merges() -> None:
    tokenizer = quality.build_bpe_prompt_tokenizer(
        gguf_tokens=[
            "W",
            "h",
            "a",
            "t",
            "Wh",
            "at",
            "What",
            "Ġ",
            "i",
            "s",
            "is",
            "Ġis",
            "?",
            "<unk>",
        ],
        gguf_merges=[
            "W h",
            "a t",
            "Wh at",
            "i s",
            "Ġ is",
        ],
        unk_token="<unk>",
    )

    assert quality.encode_prompt_text_bpe("What is?", tokenizer=tokenizer) == [
        6,
        11,
        12,
    ]


def test_quality_bpe_prompt_encoder_preserves_registered_special_tokens() -> None:
    tokenizer = quality.build_bpe_prompt_tokenizer(
        gguf_tokens=[
            "H",
            "i",
            "Hi",
            "<｜User｜>",
        ],
        gguf_merges=[
            "H i",
        ],
        special_tokens=["<｜User｜>"],
        unk_token=None,
    )

    assert quality.encode_prompt_text_bpe("<｜User｜>Hi", tokenizer=tokenizer) == [
        3,
        2,
    ]


def test_quality_renders_chat_template_for_user_prompt() -> None:
    rendered = quality.render_chat_prompt(
        "What is the capital of France?",
        chat_template="{{ bos_token }}<｜User｜>{{ messages[0]['content'] }}"
        "<｜Assistant｜></think>",
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
    )

    assert rendered == (
        "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France?"
        "<｜Assistant｜></think>"
    )


def test_quality_topk_records_decode_text() -> None:
    records = quality.topk_records(
        torch_logits=[0.1, 5.0, 1.0],
        gguf_tokens=["<s>", "Paris", " London"],
        k=2,
    )

    assert records == [
        {"token_id": 1, "text": "Paris", "logit": 5.0},
        {"token_id": 2, "text": "London", "logit": 1.0},
    ]


def test_quality_main_payload_includes_performance_metrics(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    json_path = tmp_path / "quality.json"
    times = iter([10.0, 10.5])

    monkeypatch.setattr(
        quality,
        "parse_args",
        lambda: Namespace(
            model=model_path,
            context_length=128,
            prompt_text="Hi",
            raw_prompt=True,
            prompt_token_ids="1,2",
            max_tokens=2,
            min_output_chars=1,
            json_out=json_path,
            dump_step_json=None,
            top_k=2,
            enable_profiler=False,
        ),
    )
    monkeypatch.setattr(
        quality,
        "read_gguf_tokenizer_metadata",
        lambda _path: quality.GGUFTokenizerMetadata(
            tokens=["<s>", "Hi", " there", "!"],
            merges=[],
            bos_token_id=None,
            eos_token_id=None,
            add_bos_token=False,
            chat_template=None,
        ),
    )
    monkeypatch.setattr(quality, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        quality,
        "_run_direct_generate",
        lambda *_args, **_kwargs: {
            "output_token_ids": [1, 2, 3, 2],
            "step_records": [],
            "gpu_backend": {},
            "gpu_staging": {"prefetch_payload_hits": 1},
        },
    )

    assert quality.main() == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert stdout_payload["performance"] == file_payload["performance"]
    assert stdout_payload["gpu_staging"] == {"prefetch_payload_hits": 1}
    assert stdout_payload["performance"] == {
        "generated_tokens": 2,
        "metric_scope": "direct_generation_total",
        "total_elapsed_ms": 500.0,
        "decode_tokens_per_second": 4.0,
    }


def test_quality_main_payload_includes_repeat_performance_summary(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    call_max_tokens: list[int] = []
    times = iter([10.0, 10.5, 20.0, 21.0])

    monkeypatch.setattr(
        quality,
        "parse_args",
        lambda: Namespace(
            model=model_path,
            context_length=128,
            prompt_text="Hi",
            raw_prompt=True,
            prompt_token_ids="1,2",
            max_tokens=2,
            warmup_tokens=1,
            repeat=2,
            min_decode_tps=0.0,
            max_total_elapsed_ms=0.0,
            min_output_chars=1,
            json_out=None,
            dump_step_json=None,
            top_k=2,
            enable_profiler=False,
        ),
    )
    monkeypatch.setattr(
        quality,
        "read_gguf_tokenizer_metadata",
        lambda _path: quality.GGUFTokenizerMetadata(
            tokens=["<s>", "Hi", " there", "!"],
            merges=[],
            bos_token_id=None,
            eos_token_id=None,
            add_bos_token=False,
            chat_template=None,
        ),
    )
    monkeypatch.setattr(quality, "perf_counter", lambda: next(times))

    def fake_generate(args: Namespace, *_args, **_kwargs) -> dict[str, object]:
        call_max_tokens.append(args.max_tokens)
        return {
            "output_token_ids": [1, 2, 3, 2],
            "step_records": [],
            "gpu_backend": {},
        }

    monkeypatch.setattr(quality, "_run_direct_generate", fake_generate)

    assert quality.main() == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    assert call_max_tokens == [1, 2, 2]
    assert stdout_payload["performance"] == {
        "generated_tokens": 2,
        "metric_scope": "direct_generation_total",
        "total_elapsed_ms": 500.0,
        "decode_tokens_per_second": 4.0,
    }
    assert stdout_payload["performance_summary"] == {
        "repeat": 2,
        "decode_tps_values": [4.0, 2.0],
        "decode_tps_min": 2.0,
        "decode_tps_median": 3.0,
        "decode_tps_max": 4.0,
    }


def test_quality_main_fails_explicit_decode_tps_gate(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    times = iter([10.0, 11.0])

    monkeypatch.setattr(
        quality,
        "parse_args",
        lambda: Namespace(
            model=model_path,
            context_length=128,
            prompt_text="Hi",
            raw_prompt=True,
            prompt_token_ids="1,2",
            max_tokens=2,
            warmup_tokens=0,
            repeat=1,
            min_decode_tps=3.0,
            max_total_elapsed_ms=0.0,
            min_output_chars=1,
            json_out=None,
            dump_step_json=None,
            top_k=2,
            enable_profiler=False,
        ),
    )
    monkeypatch.setattr(
        quality,
        "read_gguf_tokenizer_metadata",
        lambda _path: quality.GGUFTokenizerMetadata(
            tokens=["<s>", "Hi", " there", "!"],
            merges=[],
            bos_token_id=None,
            eos_token_id=None,
            add_bos_token=False,
            chat_template=None,
        ),
    )
    monkeypatch.setattr(quality, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        quality,
        "_run_direct_generate",
        lambda *_args, **_kwargs: {
            "output_token_ids": [1, 2, 3, 2],
            "step_records": [],
            "gpu_backend": {},
        },
    )

    assert quality.main() == 1

    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["readability"]["passed"] is True
    assert stdout_payload["performance_summary"]["decode_tps_min"] == 2.0
    assert stdout_payload["performance_gates"] == {
        "passed": False,
        "reasons": ["decode_tps_below_min"],
        "min_decode_tps": 3.0,
        "max_total_elapsed_ms": 0.0,
    }


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
