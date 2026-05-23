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


def test_production_engine_only_reads_fastinference_profile() -> None:
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
    allowed = {"FASTINFERENCE_PROFILE"}
    for rel in production_files:
        text = _read(rel)
        if rel == "vllm/engine/inference_config.py":
            text = text.split("    @classmethod\n    def from_env", maxsplit=1)[0]
        names = set(re.findall(r"FASTINFERENCE_[A-Z0-9_]+", text))
        if rel == "vllm/engine/runtime_config.py":
            assert "_FASTINFERENCE_TUNING_ENV_PREFIX" in text
            assert "_collect_temporary_fastinference_tuning_env" in text
            names.discard("FASTINFERENCE_TUNING_ENV_PREFIX")
            if "FASTINFERENCE_" in names:
                assert '_FASTINFERENCE_TUNING_ENV_PREFIX = "FASTINFERENCE_"' in text
                names.remove("FASTINFERENCE_")
        unexpected = names - allowed
        assert not unexpected, f"{rel} has production env reads: {sorted(unexpected)}"


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


def test_model_files_do_not_read_production_policy_env_names() -> None:
    production_model_files = [
        "vllm/model_executor/models/gemma4.py",
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
        text = _read(rel)
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
        "vllm/model_executor/models/gemma4.py": {
            "_GEMMA4_ALLOWED_TUNING_ENV",
            "FASTINFERENCE_GEMMA4_LAYER_PROFILE",
            "FASTINFERENCE_GEMMA4_ROCTX_PROFILE",
        },
        "vllm/model_executor/models/qwen3_5.py": {
            "_QWEN35_ALLOWED_TUNING_ENV",
        },
    }
    forbidden_patterns = (
        'key.startswith("FASTINFERENCE_GEMMA4_")',
        'key.startswith("FASTINFERENCE_KV_MAX_")',
        'key.startswith("FASTINFERENCE_QWEN35_")',
        'key.startswith("FASTINFERENCE_DISABLE_")',
    )
    for rel, required_markers in expectations.items():
        text = _read(rel)
        for marker in required_markers:
            assert marker in text, f"{rel} must document narrow tuning marker {marker}"
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{rel} must not capture tuning env by broad prefix {pattern!r}"
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
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = (
        '_env_truthy_default_on("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 local decode runtime policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_full_decode_reference_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

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
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = (
        'return _env_truthy("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 legacy full-precision KV write policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_legacy_item_path_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = ('_env_truthy("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH")',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 legacy item-path decode policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_mlp_pair_fusion_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = (
        '_env_truthy_default_on("FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION")',
    )
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 MLP pair-fusion policy must come from "
            f'attn_metadata["config"], not env-derived instance state via {pattern!r}'
        )


def test_gemma4_fp32_residual_guard_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

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
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = ('_env_int("FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE"',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 MoE expert cache policy must come from RuntimeConfig, "
            f"not env-derived instance state via {pattern!r}"
        )


def test_gemma4_awq_fused_gate_up_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

    forbidden_patterns = ('_env_truthy("FASTINFERENCE_AWQ_FUSED_GATE_UP")',)
    for pattern in forbidden_patterns:
        assert pattern not in gemma, (
            "Gemma4 AWQ fused gate-up policy must come from runtime config, "
            f"not direct env reads via {pattern!r}"
        )


def test_gemma4_rope_cache_policy_uses_runtime_config() -> None:
    gemma = _read("vllm/model_executor/models/gemma4.py")

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
    gemma = _read("vllm/model_executor/models/gemma4.py")

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
