# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _read_package(dir_path: str) -> str:
    """Read all .py files in a package directory and concatenate them."""
    pkg_dir = ROOT / dir_path
    parts = []
    for f in sorted(pkg_dir.iterdir()):
        if f.suffix == ".py":
            parts.append(f.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_lite_runtime_sources_are_parseable() -> None:
    invalid: list[str] = []
    for path in (ROOT / "vllm").rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            invalid.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not invalid, "\n".join(invalid)


def _awq_env_context_violations(
    path: str,
    *,
    prefixes: tuple[str, ...],
    allowed_defs: set[str],
) -> list[str]:
    text = _read(path)
    current_def = "<module>"
    violations: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        match = re.match(r"def ([A-Za-z0-9_]+)\(", stripped)
        if match:
            current_def = match.group(1)
        if not any(prefix in line for prefix in prefixes):
            continue
        if stripped.startswith("#"):
            continue
        if current_def in allowed_defs or current_def.endswith("_tool_override"):
            continue
        if stripped.startswith("_AWQ_FUSED_TUNING") or stripped.startswith(
            "_AWQ_TENSOR_TUNING"
        ):
            continue
        if 'key.startswith("FASTINFERENCE_AWQ_")' in line:
            continue
        if 'key.startswith("FASTINFERENCE_GEMMA4_DENSE_")' in line:
            continue
        violations.append(f"{path}:{lineno}:{current_def}:{stripped}")
    return violations


def test_fastinference_env_names_are_registered() -> None:
    from vllm.engine.env_registry import FASTINFERENCE_ENV_REGISTRY

    ignored_roots = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".gemini",
        ".claude",
        ".cursor",
        ".superpowers",
    }
    pattern = re.compile(r"(?<![A-Z0-9_])FASTINFERENCE_[A-Z0-9_]*[A-Z0-9](?![A-Z0-9_])")
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in {"tests", "vllm"}:
            continue
        if any(part in ignored_roots for part in rel.parts):
            continue
        if rel.parts[:2] == ("tests", "reports"):
            continue
        if path.suffix not in {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found.update(pattern.findall(text))

    found.discard("FASTINFERENCE_TUNING_ENV_PREFIX")
    found.discard("FASTINFERENCE_ENV_REGISTRY")
    unregistered = sorted(found - set(FASTINFERENCE_ENV_REGISTRY))
    assert not unregistered, "Unregistered FASTINFERENCE_* names: " + ", ".join(
        unregistered
    )


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


def test_broken_upstream_residue_paths_are_absent() -> None:
    removed_files = [
        "vllm/assets/__init__.py",
        "vllm/assets/audio.py",
        "vllm/assets/base.py",
        "vllm/assets/image.py",
        "vllm/assets/video.py",
        "vllm/_bc_linter.py",
        "vllm/beam_search.py",
        "vllm/cudagraph_dispatcher.py",
        "vllm/collect_env.py",
        "vllm/config/device.py",
        "vllm/config/utils.py",
        "vllm/device_allocator/cumem.py",
        "vllm/env_override.py",
        "vllm/inputs/preprocess.py",
        "vllm/logits_process.py",
        "vllm/logprobs.py",
        "vllm/metrics/perf.py",
        "vllm/metrics/stats.py",
        "vllm/model_executor/models/transformers/__init__.py",
        "vllm/model_executor/models/transformers/base.py",
        "vllm/model_executor/models/transformers/causal.py",
        "vllm/model_executor/models/transformers/legacy.py",
        "vllm/model_executor/models/transformers/moe.py",
        "vllm/model_executor/models/transformers/multimodal.py",
        "vllm/model_executor/models/transformers/pooling.py",
        "vllm/model_executor/models/transformers/utils.py",
        "vllm/model_inspection.py",
        "vllm/model_executor/models/config.py",
        "vllm/model_executor/layers/attention/kv_transfer_utils.py",
        "vllm/model_executor/layers/fused_moe/prepare_finalize.py",
        "vllm/model_executor/layers/fused_moe/shared_fused_moe.py",
        "vllm/model_executor/layers/logits_processor.py",
        "vllm/model_executor/layers/vocab_parallel_embedding.py",
        "vllm/model_executor/models/mixtral.py",
        "vllm/model_executor/models/qwen2_moe.py",
        "vllm/model_executor/model_loader/weight_utils.py",
        "vllm/attention/backend.py",
        "vllm/forward_context.py",
        "vllm/entrypoints/openai/engine/serving.py",
        "vllm/multimodal/audio.py",
        "vllm/multimodal/media/audio.py",
        "vllm/multimodal/media/video.py",
        "vllm/multimodal/parse.py",
        "vllm/multimodal/video.py",
        "vllm/multimodal/registry.py",
        "vllm/renderers/mistral.py",
        "vllm/renderers/deepseek_v32.py",
        "vllm/serial_utils.py",
        "vllm/request.py",
        "vllm/scalar_type.py",
        "vllm/sequence.py",
        "vllm/tasks.py",
        "vllm/tokenizers/mistral.py",
        "vllm/tokenizers/deepseek_v32.py",
        "vllm/tracing.py",
        "vllm/transformers_utils/config_parser_base.py",
        "vllm/transformers_utils/processor.py",
        "vllm/v1_outputs.py",
        "vllm/v1_utils.py",
        "vllm/transformers_utils/configs/qwen3_next.py",
        "vllm/transformers_utils/model_arch_config_convertor.py",
        "vllm/transformers_utils/processors/hunyuan_vl_image.py",
    ]

    present = [path for path in removed_files if (ROOT / path).exists()]
    assert not present, "Broken upstream residue paths must stay removed: " + ", ".join(
        present
    )


def test_lite_runtime_does_not_import_removed_upstream_runtimes() -> None:
    forbidden = re.compile(
        r"vllm\.(distributed|worker|core|executor|grpc|spec_decode|compilation)"
    )
    matches = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "vllm").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert not matches, "Removed runtime imports must stay absent: " + ", ".join(
        matches
    )


def test_readme_documents_runtime_profiles_instead_of_legacy_runtime_env_matrix() -> (
    None
):
    readme = _read("README.md")

    required = (
        "Runtime Profiles",
        "tuning_keyvals",
        "benchmark",
        "latency",
        "throughput",
        "accuracy",
    )
    for marker in required:
        assert marker in readme, f"README must document runtime profiles via {marker!r}"

    forbidden = (
        "## 🛠️ 配置指南 (LiteInferenceConfig)",
        "FASTINFERENCE_PROFILE",
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_FUSION_LEVEL",
    )
    for marker in forbidden:
        assert marker not in readme, (
            "README runtime documentation must no longer present legacy env "
            f"matrix marker {marker!r}"
        )


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


def test_lite_backend_does_not_read_fastinference_env_directly() -> None:
    backend = _read("vllm/engine/backend/lite_single_gpu.py")

    forbidden_patterns = (
        "os.environ",
        "getenv",
        "FASTINFERENCE_",
    )
    for pattern in forbidden_patterns:
        assert pattern not in backend, (
            "lite backend must receive runtime policy from BackendRuntimePolicy, "
            f"not read env directly via {pattern!r}"
        )


def test_request_builder_does_not_read_fastinference_env_directly() -> None:
    builder = _read("vllm/engine/request_builder.py")

    forbidden_patterns = (
        "os.environ",
        "getenv",
        "FASTINFERENCE_",
    )
    for pattern in forbidden_patterns:
        assert pattern not in builder, (
            "request builder must receive runtime policy from RuntimeConfig, "
            f"not read env directly via {pattern!r}"
        )


def test_lite_engine_operational_policy_does_not_read_env_directly() -> None:
    engine = _read("vllm/engine/lite_engine.py")

    forbidden_patterns = (
        'os.environ.get("FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS"',
        'os.environ.get("FASTINFERENCE_MEM_AUDIT_TOPN"',
    )
    for pattern in forbidden_patterns:
        assert pattern not in engine, (
            "lite engine operational policy must come from RuntimeConfig, "
            f"not direct env read via {pattern!r}"
        )


def test_production_engine_uses_config_not_public_runtime_env_controls() -> None:
    production_files = [
        "vllm/engine/runtime_profile.py",
        "vllm/engine/runtime_config.py",
        "vllm/engine/inference_config.py",
        "vllm/engine/lite_engine.py",
        "vllm/engine/step_scheduler.py",
        "vllm/engine/runtime_factory.py",
        "vllm/engine/runtime_controller.py",
        "vllm/engine/backend/lite_single_gpu.py",
    ]
    allowed = {"FASTINFERENCE_CONFIG"}
    for rel in production_files:
        text = _read(rel)
        if rel == "vllm/engine/inference_config.py":
            text = text.split("    @classmethod\n    def from_env", maxsplit=1)[0]
        names = set(re.findall(r"FASTINFERENCE_[A-Z0-9_]+", text))
        if rel == "vllm/engine/runtime_config.py":
            assert "resolve_fastinference_config" in text
            assert "_collect_temporary_fastinference_tuning_env" not in text
            assert "_FASTINFERENCE_TUNING_ENV_PREFIX" not in text
        file_allowed = set(allowed)
        if rel == "vllm/engine/runtime_profile.py":
            file_allowed.add("FASTINFERENCE_PROFILE")
        if rel == "vllm/engine/runtime_config.py":
            file_allowed.add("FASTINFERENCE_PROFILE")
            file_allowed.update(
                {
                    "FASTINFERENCE_ALLOW_LEGACY_ENV",
                    "FASTINFERENCE_BENCH_PROFILE",
                    "FASTINFERENCE_DEBUG",
                    "FASTINFERENCE_KV_TYPE",
                    "FASTINFERENCE_LOG_LEVEL",
                    "FASTINFERENCE_USE_LEGACY_SAMPLING",
                }
            )
        unexpected = names - file_allowed
        assert not unexpected, f"{rel} has production env reads: {sorted(unexpected)}"


def test_runtime_profile_has_config_resolver_for_production_path() -> None:
    profile = _read("vllm/engine/runtime_profile.py")
    production_text = profile.split(
        "    @classmethod\n    def resolve_from_env", maxsplit=1
    )[0]

    assert "resolve_from_config" in profile
    assert 'os.environ.get("FASTINFERENCE_PROFILE"' not in production_text


def test_production_step_scheduler_does_not_call_inference_config_from_env() -> None:
    scheduler = _read("vllm/engine/step_scheduler.py")

    forbidden_patterns = (
        "LiteInferenceConfig",
        "from_env()",
    )
    for pattern in forbidden_patterns:
        assert pattern not in scheduler, (
            "production StepScheduler must receive prefill policy from "
            f"RuntimeConfig, not LiteInferenceConfig.from_env via {pattern!r}"
        )


def test_lite_inference_config_has_no_env_factory() -> None:
    inference_config = _read("vllm/engine/inference_config.py")

    forbidden_patterns = (
        "def from_env",
        'os.environ.get("FASTINFERENCE_',
        "os.environ.items()",
    )
    for pattern in forbidden_patterns:
        assert pattern not in inference_config, (
            "LiteInferenceConfig must be populated from RuntimeConfig/config "
            f"rather than hidden env controls via {pattern!r}"
        )


def test_kv_attention_layers_do_not_read_fastinference_env_directly() -> None:
    production_files = [
        "vllm/model_executor/layers/quantization/kv_cache.py",
    ]
    forbidden_patterns = (
        'os.environ.get("FASTINFERENCE_',
        'os.getenv("FASTINFERENCE_',
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_K_SCALE",
        "FASTINFERENCE_V_SCALE",
    )
    for rel in production_files:
        text = _read(rel)
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{rel} must receive KV policy from RuntimeConfig/cache metadata, "
                f"not hidden env controls via {pattern!r}"
            )


def test_awq_kernels_do_not_capture_fastinference_env_by_prefix() -> None:
    production_files = [
        "vllm/kernels/triton/awq_fused_gemm.py",
        "vllm/model_executor/layers/quantization/tensor.py",
    ]
    forbidden_patterns = (
        "os.environ.items()",
        'key.startswith("FASTINFERENCE_AWQ_")',
        'key.startswith("FASTINFERENCE_GEMMA4_DENSE_")',
    )
    for rel in production_files:
        text = _read(rel)
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{rel} must not capture arbitrary tuning env by prefix via {pattern!r}"
            )


def test_production_runtime_has_no_direct_fastinference_env_reads() -> None:
    ignored = {
        "vllm/engine/env_registry.py",
        "vllm/engine/fastinference_config.py",
        "vllm/engine/runtime_config.py",
        "vllm/engine/runtime_profile.py",
    }
    pattern = re.compile(r"os\.environ\.get\(\"(FASTINFERENCE_[A-Z0-9_]+)\"")
    violations: list[str] = []
    for path in (ROOT / "vllm").rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if rel in ignored or "__pycache__" in rel:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = pattern.search(line)
            if match:
                violations.append(f"{rel}:{lineno}:{match.group(1)}")
    assert not violations, (
        "Production runtime must not directly read FASTINFERENCE_* env outside "
        "the config/registry bridge: " + ", ".join(violations)
    )


def test_model_files_do_not_read_production_policy_env_names() -> None:
    production_model_files = [
        "vllm/model_executor/models/gemma4/",
        "vllm/model_executor/models/qwen3_5.py",
    ]
    forbidden_names = {
        "FASTINFERENCE_AWQ_FUSED_GATE_UP",
        "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN",
        "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN",
        "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE",
        "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH",
        "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON",
        "FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION",
        "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER",
        "FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL",
        "FASTINFERENCE_QWEN35_USE_FLA_CHUNK",
        "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP",
        "FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER",
    }
    forbidden_patterns = ("os.environ.get(", "getenv(")
    for rel in production_model_files:
        text = _read_package(rel) if rel.endswith("/") else _read(rel)
        found_names = forbidden_names & set(
            re.findall(r"FASTINFERENCE_[A-Z0-9_]+", text)
        )
        assert not found_names, (
            f"{rel} must receive production model/kernel policy from adapters, "
            f"not direct env/tuning names: {sorted(found_names)}"
        )
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{rel} must not read production policy directly via {pattern!r}"
            )


def test_model_tuning_snapshots_use_narrow_allowlists() -> None:
    expectations = {
        "vllm/model_executor/models/gemma4/": {
            "_GEMMA4_ALLOWED_TUNING_ENV",
            "FASTINFERENCE_GEMMA4_LAYER_PROFILE",
            "FASTINFERENCE_GEMMA4_ROCTX_PROFILE",
        }
    }
    forbidden_patterns = (
        'key.startswith("FASTINFERENCE_GEMMA4_")',
        'key.startswith("FASTINFERENCE_KV_MAX_")',
        'key.startswith("FASTINFERENCE_QWEN35_")',
        'key.startswith("FASTINFERENCE_DISABLE_")',
    )
    for rel, required_markers in expectations.items():
        text = _read_package(rel) if rel.endswith("/") else _read(rel)
        for marker in required_markers:
            assert marker in text, f"{rel} must document narrow tuning marker {marker}"
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{rel} must not capture tuning env by broad prefix {pattern!r}"
            )


def test_qwen35_model_has_no_tuning_snapshot_state() -> None:
    qwen = _read("vllm/model_executor/models/qwen3_5.py")

    forbidden_patterns = (
        "_QWEN35_TUNING",
        "_QWEN35_ALLOWED_TUNING_ENV",
        "set_qwen35_tuning_config",
        "os.environ.items()",
    )
    for pattern in forbidden_patterns:
        assert pattern not in qwen, (
            "Qwen3.5 production policy is adapter-owned; model-local tuning "
            f"snapshot state must not remain via {pattern!r}"
        )


def test_qwen35_full_attention_policy_uses_runtime_config() -> None:
    qwen = _read("vllm/model_executor/models/qwen3_5.py")

    forbidden_patterns = (
        "self._use_full_attn_stabilizer = _env_truthy(",
        "if self._use_full_attn_stabilizer:",
        "self._use_sdpa_prefill = _env_qwen35_sdpa_prefill_enabled()",
        "self._use_sdpa_prefill",
        '_env_truthy("FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER")',
        '_env_truthy("FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in qwen, (
            "Qwen3.5 full-attention runtime policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_local_decode_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_env_truthy_default_on("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 local decode runtime policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_full_decode_reference_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_env_truthy("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN")',
        '_env_truthy("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 full-decode reference policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_legacy_full_precision_kv_write_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        'return _env_truthy("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 legacy full-precision KV write policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_legacy_item_path_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = ('_env_truthy("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH")',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 legacy item-path decode policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_mlp_pair_fusion_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_env_truthy_default_on("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 MLP pair-fusion policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_fp32_residual_guard_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_env_truthy("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD")',
        '_env_int_alias("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START"',
        '_env_int("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN"',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 fp32 residual guard policy must come from RuntimeConfig, "
            f"not env-derived instance state via {pattern!r}"
        )


def test_gemma4_moe_expert_cache_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = ('_env_int("FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE"',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 MoE expert cache policy must come from RuntimeConfig, "
            f"not env-derived instance state via {pattern!r}"
        )


def test_gemma4_awq_fused_gate_up_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = ('_env_truthy("FASTINFERENCE_AWQ_FUSED_GATE_UP")',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 AWQ fused gate-up policy must come from runtime config, "
            f"not direct env reads via {pattern!r}"
        )


def test_gemma4_rope_cache_policy_uses_runtime_config() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_env_get("FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS"',
        '_env_get("FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX"',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 rope cache policy must come from RuntimeConfig, "
            f"not env-derived instance state via {pattern!r}"
        )


def test_gemma4_profile_flags_do_not_read_env_at_module_init() -> None:
    gemma = _read_package("vllm/model_executor/models/gemma4/")

    forbidden_patterns = (
        '_GEMMA4_PROFILE_ENABLED = _env_get("FASTINFERENCE_GEMMA4_LAYER_PROFILE"',
        '_GEMMA4_ROCTX_PROFILE_ENABLED = _env_get("FASTINFERENCE_GEMMA4_ROCTX_PROFILE"',
        '_env_truthy("FASTINFERENCE_GEMMA4_LAYER_PROFILE")',
        '_env_truthy("FASTINFERENCE_GEMMA4_ROCTX_PROFILE")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 profile flags must be installed from tuning config, "
            f"not module-init env reads via {pattern!r}"
        )


def test_awq_tensor_production_env_reads_are_tool_only() -> None:
    violations = _awq_env_context_violations(
        "vllm/model_executor/layers/quantization/tensor.py",
        prefixes=(
            "FASTINFERENCE_AWQ_FUSED_GEMM",
            "FASTINFERENCE_AWQ_FUSED_GEMM_FORCE",
            "FASTINFERENCE_AWQ_DECODE_GEMV",
            "FASTINFERENCE_AWQ_CACHE_SCOPE",
            "FASTINFERENCE_AWQ_POLICY_MATRIX",
            "FASTINFERENCE_AWQ_FUSED_SCOPE",
            "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ",
            "FASTINFERENCE_GEMMA4_DENSE_MLP",
        ),
        allowed_defs={
            "_env_get",
            "_env_truthy",
            "set_awq_tensor_tuning_config",
        },
    )

    assert not violations, (
        "tensor.py production AWQ policy must come from kernel_policy/defaults; "
        "migrated FASTINFERENCE_AWQ_* and FASTINFERENCE_GEMMA4_DENSE_* names are only "
        "allowed in tool-only helpers or tuning snapshot code:\n"
        + "\n".join(violations)
    )


def test_awq_fused_gemm_production_env_reads_are_tool_only() -> None:
    violations = _awq_env_context_violations(
        "vllm/kernels/triton/awq_fused_gemm.py",
        prefixes=("FASTINFERENCE_AWQ_",),
        allowed_defs={
            "_env_get",
            "set_awq_fused_tuning_config",
            "_persistent_profile_path",
            "_fused_gemm_split_k_snapshot_override",
        },
    )

    assert not violations, (
        "awq_fused_gemm.py production launch policy must come from "
        "kernel_policy/defaults plus persistent profile lookups; "
        "FASTINFERENCE_AWQ_* names are only allowed in tool-only helpers, "
        "tuning snapshot code, locked snapshot split-k helper, or "
        "_persistent_profile_path:\n" + "\n".join(violations)
    )
