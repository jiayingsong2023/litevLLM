# Runtime Profile Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace production `FASTINFERENCE_*` strategy branches with deterministic runtime profiles while preserving benchmark/tool tuning controls and requiring accuracy plus performance reports after every phase.

**Architecture:** Add a small profile registry under `vllm/engine/` that resolves `FASTINFERENCE_PROFILE` into typed scheduler, backend, KV, model, and kernel policies. `RuntimeConfig` becomes a consumer of resolved profile data instead of a broad environment parser; adapters own model-family policy, and production modules receive policy objects through existing `attn_metadata["config"]` and runtime assembly.

**Tech Stack:** Python 3.12, dataclasses, existing `RuntimeConfig`, `SchedulerRuntimePolicy`, `BackendRuntimePolicy`, pytest, existing regression scripts, existing benchmark tools.

---

## File Structure

- Create: `vllm/engine/runtime_profile.py`
  - Owns `RuntimeProfile`, `RuntimeProfileRegistry`, profile names, effective-policy snapshots, and the allowlisted `FASTINFERENCE_PROFILE` read.
- Create: `vllm/engine/runtime_policy.py`
  - Owns `SchedulerRuntimePolicy` and `BackendRuntimePolicy` so profile and config modules do not depend on each other.
- Modify: `vllm/engine/runtime_config.py`
  - Replaces broad engine/backend/scheduler env parsing with profile resolution.
- Modify: `vllm/engine/runtime_factory.py`
  - Imports scheduler/backend policy dataclasses from `runtime_policy.py`.
- Modify: `vllm/engine/runtime_controller.py`
  - Includes resolved profile metadata in `stats()`.
- Modify: `vllm/engine/lite_engine.py`
  - Removes stale helper env reads and reports resolved profile during startup.
- Modify: `vllm/adapters/base.py`
  - Extends `RuntimeModelPolicy` only where needed to carry stable model policy without adding generic Gemma4 fields to `RuntimeConfig`.
- Modify: `vllm/adapters/gemma4.py`, `vllm/adapters/qwen3_5.py`
  - Move production model-family decisions into adapter policies.
- Modify: `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/qwen3_5.py`
  - Stop reading production model policy from env/tuning keys; consume config/profile policy.
- Modify: `vllm/model_executor/layers/quantization/tensor.py`, `vllm/kernels/triton/awq_fused_gemm.py`
  - Split production kernel policy from tool-only tuning overrides.
- Modify: `tests/test_kv_default_policy.py`
  - Replace env-ownership tests with profile-ownership tests.
- Create: `tests/test_runtime_profile.py`
  - Unit tests for profile resolution and effective policy values.
- Modify: `tests/test_runtime_stats_export.py`
  - Assert runtime stats include profile metadata.
- Modify: `tests/test_project_governance.py`
  - Add production `FASTINFERENCE_*` guardrails with explicit allowlist.
- Modify: `tests/e2e_full_benchmark.py`
  - Report resolved profile and effective policy values in benchmark output.
- Modify: `README.md`, `docs/ARCHITECTURE_LITE.md`, `docs/CAPABILITY_MATRIX.md`
  - Replace large environment-variable guidance with profile guidance.

## Phase Reports Required After Every Task

Each task ends with a short report in the implementation response:

```text
Accuracy report:
- Commands:
- Result:
- Coverage:
- Skips:

Performance report:
- Profile:
- Model:
- Shape:
- Metrics:
- Baseline comparison:
- Skips:
```

For tasks that do not change runtime behavior, performance report may say
`Skipped: no runtime behavior change` and must still include the intended profile
coverage for the next runtime-affecting task.

---

### Task 1: Add Runtime Profile Data Model

**Files:**
- Create: `vllm/engine/runtime_policy.py`
- Create: `vllm/engine/runtime_profile.py`
- Test: `tests/test_runtime_profile.py`

- [ ] **Step 1: Write failing profile resolution tests**

Create `tests/test_runtime_profile.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.runtime_profile import (
    RuntimeProfileRegistry,
    SUPPORTED_PROFILE_NAMES,
)


def _caps(model_type: str = "llama") -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        supports_moe=False,
        supports_int4_kv=True,
        supports_fp8_kv=True,
        max_model_len=4096,
    )


def test_supported_profile_names_are_stable() -> None:
    assert SUPPORTED_PROFILE_NAMES == (
        "auto",
        "latency",
        "throughput",
        "accuracy",
        "benchmark",
    )


def test_auto_profile_resolves_to_named_effective_profile(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    profile = RuntimeProfileRegistry.resolve(
        requested_profile=None,
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES
    assert profile.kv_cache_dtype == "turbo_int4"
    assert profile.block_size == 16
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_env_profile_is_the_only_fastinference_runtime_selector(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")

    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "accuracy"
    assert profile.kv_cache_dtype == "fp8"


def test_unknown_profile_falls_back_to_auto(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "experimental_local")
    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_profile.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.engine.runtime_profile'`.

- [ ] **Step 3: Implement profile model**

Create `vllm/engine/runtime_policy.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerRuntimePolicy:
    max_decode_streak: int = 4
    queue_aging_threshold_s: float = 2.0
    max_prefill_deferrals: int = 2
    service_class_weights: dict[str, int] | None = None
    admission_service_class_quotas: dict[str, int] | None = None
    decode_service_class_quotas: dict[str, int] | None = None
    fairness_guardrail_queue_wait_s: float = 0.0
    fairness_guardrail_service_classes: set[str] | None = None
    max_admit_lora_adapters_per_step: int = 0
    max_prefill_lora_adapters_per_batch: int = 0
    max_decode_lora_adapters_per_batch: int = 0
    lora_fairness_relax_threshold: float = 0.0
    lora_locality_tighten_threshold: float = 0.0
    lora_limit_relax_delta: int = 1
    lora_limit_tighten_delta: int = 1
    max_admit_multimodal_per_step: int = 0
    max_prefill_multimodal_requests_per_batch: int = 0
    max_decode_multimodal_requests_per_batch: int = 0
    max_admit_multimodal_lora_per_step: int = 0
    max_prefill_multimodal_lora_requests_per_batch: int = 0
    max_decode_multimodal_lora_requests_per_batch: int = 0
    multimodal_prefix_cache_relax_threshold: float = 0.0
    multimodal_prefix_cache_tighten_threshold: float = 0.0
    multimodal_prefill_limit_relax_delta: int = 1
    multimodal_prefill_limit_tighten_delta: int = 1
    multimodal_lora_prefill_limit_relax_delta: int = 1
    multimodal_lora_prefill_limit_tighten_delta: int = 1
    multimodal_lora_fairness_relax_threshold: float = 0.0
    multimodal_lora_locality_tighten_threshold: float = 0.0


@dataclass(frozen=True)
class BackendRuntimePolicy:
    max_prefix_cache_entries: int = 8
    preemption_mode: str = "defer_prefill"
    preemption_min_backlog: int = 1
    preemption_min_decodes: int = 1
    preemption_max_queue_wait_s: float = 0.0
    preemptible_service_classes: set[str] | None = None
    preempt_multimodal_prefills: bool = False
    preempt_multimodal_max_queue_wait_s: float = 0.0
    multimodal_prefix_cache_protect_threshold: float = 0.0
    gpu_greedy_sampling: bool = False
    gpu_greedy_max_tokens_only: bool = False
    gpu_greedy_bypass_cpu_policies: bool = False
    gpu_greedy_ignore_eos: bool = False
```

Create `vllm/engine/runtime_profile.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy

SUPPORTED_PROFILE_NAMES = (
    "auto",
    "latency",
    "throughput",
    "accuracy",
    "benchmark",
)


@dataclass(frozen=True)
class RuntimeProfile:
    requested_name: str
    effective_name: str
    description: str
    kv_cache_dtype: str = "turbo_int4"
    block_size: int = 16
    kv_max_model_len: int | None = None
    kv_max_active_requests: int = 4
    fusion_level: int = 2
    enable_decode_priority: bool = True
    prefill_chunk_size: int = 0
    prefill_reserved_tokens: int = 0
    prefill_reserve_backlog: int = 2
    prefill_catchup_ratio: float = 0.25
    prefill_microbatch_size: int = 2
    default_min_new_tokens: int = 0
    queue_timeout_s: float = 30.0
    memory_audit_topn: int = 20
    k_scale: float = 1.0
    v_scale: float = 1.0
    use_prompt_guard: bool = True
    scheduler_policy: SchedulerRuntimePolicy = field(
        default_factory=SchedulerRuntimePolicy
    )
    backend_policy: BackendRuntimePolicy = field(default_factory=BackendRuntimePolicy)
    model_policy: dict[str, Any] = field(default_factory=dict)
    kernel_policy: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        return {
            "requested": self.requested_name,
            "effective": self.effective_name,
            "description": self.description,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.block_size,
            "kv_max_model_len": self.kv_max_model_len,
            "kv_max_active_requests": self.kv_max_active_requests,
            "fusion_level": self.fusion_level,
        }


class RuntimeProfileRegistry:
    @classmethod
    def resolve_from_env(
        cls,
        *,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> RuntimeProfile:
        requested = os.environ.get("FASTINFERENCE_PROFILE", "auto").strip().lower()
        return cls.resolve(
            requested_profile=requested,
            model_capabilities=model_capabilities,
            gpu_total_gb=gpu_total_gb,
        )

    @classmethod
    def resolve(
        cls,
        *,
        requested_profile: str | None,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> RuntimeProfile:
        requested = (requested_profile or "auto").strip().lower()
        if requested not in SUPPORTED_PROFILE_NAMES:
            requested = "auto"
        effective = cls._effective_name(requested, model_capabilities, gpu_total_gb)
        return cls._build(requested_name=requested, effective_name=effective)

    @staticmethod
    def _effective_name(
        requested: str,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> str:
        del model_capabilities, gpu_total_gb
        if requested == "auto":
            return "benchmark"
        return requested

    @staticmethod
    def _build(*, requested_name: str, effective_name: str) -> RuntimeProfile:
        if effective_name == "accuracy":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Accuracy-first profile with conservative KV policy.",
                kv_cache_dtype="fp8",
            )
        if effective_name == "latency":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Single-request and low-TPOT profile.",
                kv_cache_dtype="turbo_int4",
                kv_max_active_requests=1,
                kv_max_model_len=512,
                backend_policy=BackendRuntimePolicy(
                    gpu_greedy_sampling=True,
                    gpu_greedy_max_tokens_only=True,
                    gpu_greedy_bypass_cpu_policies=True,
                ),
            )
        if effective_name == "throughput":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Batch throughput profile for supported dense shapes.",
                kv_cache_dtype="turbo_int4",
                kv_max_active_requests=16,
                prefill_microbatch_size=4,
            )
        return RuntimeProfile(
            requested_name=requested_name,
            effective_name=effective_name,
            description="Benchmark-recommended stable production profile.",
            kv_cache_dtype="turbo_int4",
            backend_policy=BackendRuntimePolicy(
                gpu_greedy_sampling=True,
                gpu_greedy_max_tokens_only=True,
                gpu_greedy_bypass_cpu_policies=True,
            ),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_profile.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/engine/runtime_policy.py vllm/engine/runtime_profile.py tests/test_runtime_profile.py
git commit -m "feat: add runtime profile registry"
```

- [ ] **Step 6: Report**

Accuracy report:
- `uv run pytest tests/test_runtime_profile.py -q`

Performance report:
- Skipped: profile model only; no runtime behavior change.

---

### Task 2: Route RuntimeConfig Through RuntimeProfile

**Files:**
- Modify: `vllm/engine/runtime_policy.py`
- Modify: `vllm/engine/runtime_config.py`
- Modify: `vllm/engine/runtime_factory.py`
- Modify: `tests/test_kv_default_policy.py`

- [ ] **Step 1: Rewrite failing RuntimeConfig tests**

In `tests/test_kv_default_policy.py`, replace env-ownership tests for KV,
scheduler, backend, request-builder defaults, memory audit, and Gemma4 runtime
fields with profile assertions:

```python
def test_runtime_config_uses_profile_defaults(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.profile.requested_name == "auto"
    assert cfg.profile.effective_name == "benchmark"
    assert cfg.kv_cache_dtype == "turbo_int4"
    assert cfg.block_size == 16
    assert cfg.fusion_level == 2
    assert cfg.backend_policy.gpu_greedy_sampling is True


def test_runtime_config_accuracy_profile_is_conservative(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "turbo_int4")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "fp8"


def test_runtime_config_latency_profile_caps_kv_shape(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "latency")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.kv_max_active_requests == 1
    assert cfg.kv_max_model_len == 512
```

Keep existing `LiteInferenceConfig.from_env()` tests temporarily because
`LiteInferenceConfig` is not removed until later tasks.

- [ ] **Step 2: Run changed tests to verify failure**

Run: `uv run pytest tests/test_kv_default_policy.py -q`

Expected: FAIL because `RuntimeConfig` has no `profile` field and still honors
legacy env values.

- [ ] **Step 3: Add RuntimeConfig.profile and profile resolution**

Modify `vllm/engine/runtime_config.py` so it imports policy dataclasses from
`runtime_policy.py` and no longer defines independent copies:

```python
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy
```

Modify `vllm/engine/runtime_factory.py` to import from the new policy module:

```python
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy
```

Then import the profile registry:

```python
from vllm.engine.runtime_profile import RuntimeProfile, RuntimeProfileRegistry
```

Add field:

```python
profile: RuntimeProfile | None = None
```

At the top of `from_vllm_config()`:

```python
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb

profile = RuntimeProfileRegistry.resolve_from_env(
    model_capabilities=getattr(vllm_config, "model_capabilities", None),
    gpu_total_gb=get_total_gpu_memory_gb(),
)
```

Set runtime fields from `profile`:

```python
return cls(
    model_path=str(model_config.model),
    tokenizer_path=str(model_config.tokenizer),
    dtype=str(model_config.dtype),
    max_model_len=int(getattr(model_config, "max_model_len", 2048)),
    max_num_seqs=int(getattr(scheduler_config, "max_num_seqs", 4)),
    max_num_batched_tokens=int(
        getattr(scheduler_config, "max_num_batched_tokens", 4096)
    ),
    block_size=profile.block_size,
    kv_cache_dtype=profile.kv_cache_dtype,
    kv_max_model_len=profile.kv_max_model_len,
    kv_max_active_requests=profile.kv_max_active_requests,
    fusion_level=profile.fusion_level,
    policy_mode=str(getattr(vllm_config, "runtime_policy_mode", "auto")).lower(),
    enable_decode_priority=profile.enable_decode_priority,
    prefill_chunk_size=profile.prefill_chunk_size,
    prefill_reserved_tokens=profile.prefill_reserved_tokens,
    prefill_reserve_backlog=profile.prefill_reserve_backlog,
    prefill_catchup_ratio=profile.prefill_catchup_ratio,
    prefill_microbatch_size=profile.prefill_microbatch_size,
    default_min_new_tokens=profile.default_min_new_tokens,
    queue_timeout_s=profile.queue_timeout_s,
    memory_audit_topn=profile.memory_audit_topn,
    k_scale=profile.k_scale,
    v_scale=profile.v_scale,
    use_prompt_guard=profile.use_prompt_guard,
    tuning_env={"FASTINFERENCE_PROFILE": profile.requested_name},
    scheduler_policy=profile.scheduler_policy,
    backend_policy=profile.backend_policy,
    profile=profile,
)
```

Keep Gemma4-specific dataclass fields populated from profile defaults in this
task. Do not remove fields yet; later tasks move them into adapter policy.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_profile.py tests/test_kv_default_policy.py -q`

Expected: PASS after test updates.

- [ ] **Step 5: Commit**

```bash
git add vllm/engine/runtime_policy.py vllm/engine/runtime_config.py vllm/engine/runtime_factory.py tests/test_kv_default_policy.py
git commit -m "feat: resolve runtime config from profiles"
```

- [ ] **Step 6: Report**

Accuracy report:
- `uv run pytest tests/test_runtime_profile.py tests/test_kv_default_policy.py -q`

Performance report:
- Skipped: config source changed, effective default profile should match prior benchmark defaults for supported models.

---

### Task 3: Surface Profile Metadata In Runtime Stats And Benchmarks

**Files:**
- Modify: `vllm/engine/runtime_controller.py`
- Modify: `tests/test_runtime_stats_export.py`
- Modify: `tests/e2e_full_benchmark.py`

- [ ] **Step 1: Write failing stats test**

Update fake stats in `tests/test_runtime_stats_export.py` to include profile:

```python
def stats(self) -> dict[str, object]:
    return {
        "profile": {
            "requested": "auto",
            "effective": "benchmark",
            "kv_cache_dtype": "turbo_int4",
        },
        "scheduler": {"active_request_count": 2},
        "observer": {"step_count": 3},
    }
```

Assert the endpoint preserves that profile under `stats`.

Add a direct controller-level unit if needed:

```python
def test_runtime_controller_stats_include_profile() -> None:
    # Construct a RuntimeController with fake scheduler/backend/observer and
    # assert stats()["profile"] comes from backend or runtime config.
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_runtime_stats_export.py -q`

Expected: FAIL until runtime stats consistently include profile metadata.

- [ ] **Step 3: Add profile stats**

In `RuntimeController.stats()`, add:

```python
profile = {}
runtime_config = getattr(self.scheduler, "runtime_config", None)
if runtime_config is not None and getattr(runtime_config, "profile", None) is not None:
    profile = runtime_config.profile.stats()
elif hasattr(self.backend, "profile_stats"):
    profile = dict(self.backend.profile_stats())
```

Return it:

```python
"profile": profile,
```

If `RequestScheduler` does not carry runtime config, attach it from
`LiteEngine.__init__` after scheduler creation:

```python
self.scheduler.runtime_config = self.runtime_config
```

- [ ] **Step 4: Add benchmark report fields**

In `tests/e2e_full_benchmark.py`, when collecting engine stats, include:

```python
profile_stats = dict(runtime_stats.get("profile") or {})
report["profile"] = profile_stats
```

In terminal summary, print:

```python
print(
    "PROFILE "
    f"requested={profile_stats.get('requested', 'unknown')} "
    f"effective={profile_stats.get('effective', 'unknown')} "
    f"kv={profile_stats.get('kv_cache_dtype', 'unknown')}"
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_runtime_stats_export.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/engine/runtime_controller.py vllm/engine/lite_engine.py tests/test_runtime_stats_export.py tests/e2e_full_benchmark.py
git commit -m "feat: expose runtime profile in stats"
```

- [ ] **Step 7: Report**

Accuracy report:
- `uv run pytest tests/test_runtime_stats_export.py -q`

Performance report:
- Skipped: observability only; benchmark output now records profile identity for future performance reports.

---

### Task 4: Remove Engine-Level Production Env Policy Reads

**Files:**
- Modify: `vllm/engine/runtime_config.py`
- Modify: `vllm/engine/lite_engine.py`
- Modify: `tests/test_project_governance.py`
- Modify: `tests/test_kv_default_policy.py`

- [ ] **Step 1: Add governance allowlist test**

In `tests/test_project_governance.py`, add:

```python
def test_production_engine_only_reads_fastinference_profile() -> None:
    production_files = [
        "vllm/engine/runtime_config.py",
        "vllm/engine/lite_engine.py",
        "vllm/engine/runtime_factory.py",
        "vllm/engine/runtime_controller.py",
        "vllm/engine/backend/lite_single_gpu.py",
    ]
    allowed = {"FASTINFERENCE_PROFILE"}
    for rel in production_files:
        text = _read(rel)
        names = set(re.findall(r"FASTINFERENCE_[A-Z0-9_]+", text))
        unexpected = names - allowed
        assert not unexpected, f"{rel} has production env reads: {sorted(unexpected)}"
```

- [ ] **Step 2: Run governance test to verify failure**

Run: `uv run pytest tests/test_project_governance.py::test_production_engine_only_reads_fastinference_profile -q`

Expected: FAIL listing existing engine-level `FASTINFERENCE_*` names.

- [ ] **Step 3: Remove stale helpers and warning text**

In `vllm/engine/lite_engine.py`, delete unused helpers:

```python
def _resolve_kv_max_model_len(...): ...
def _resolve_kv_max_active_requests(...): ...
```

Replace warning text mentioning env controls:

```python
"reduce the selected runtime profile context/concurrency limits or choose the accuracy profile."
```

Replace fused stage print hint with profile language:

```python
"fused rollout stage={fused_stage} (from runtime profile/model policy)."
```

- [ ] **Step 4: Remove broad env parsing from RuntimeConfig**

Delete parsing for engine-level env controls from `from_vllm_config()`:

```python
FASTINFERENCE_KV_TYPE
FASTINFERENCE_KV_FP8
FASTINFERENCE_BLOCK_SIZE
FASTINFERENCE_KV_MAX_MODEL_LEN
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS
FASTINFERENCE_FUSION_LEVEL
FASTINFERENCE_LITE_PREFILL_*
FASTINFERENCE_MAX_DECODE_STREAK
FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES
FASTINFERENCE_PREEMPTION_*
FASTINFERENCE_GPU_GREEDY_*
FASTINFERENCE_MEM_AUDIT_TOPN
```

Keep only the `RuntimeProfileRegistry.resolve_from_env()` call.

- [ ] **Step 5: Update tests**

Remove tests that expect env ownership from `tests/test_kv_default_policy.py`.
Keep tests that assert profile behavior.

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_runtime_profile.py tests/test_kv_default_policy.py tests/test_project_governance.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/engine/runtime_config.py vllm/engine/lite_engine.py tests/test_project_governance.py tests/test_kv_default_policy.py
git commit -m "refactor: remove engine runtime env policy reads"
```

- [ ] **Step 8: Report**

Accuracy report:
- `uv run pytest tests/test_runtime_profile.py tests/test_kv_default_policy.py tests/test_project_governance.py -q`

Performance report:
- Skipped if no model run is available.
- If models are available, run one baseline profile smoke:
  `FASTINFERENCE_PROFILE=benchmark uv run python tests/e2e_full_benchmark.py --models tinyllama --warmup-preset off`

---

### Task 5: Move Gemma4 And Qwen3.5 Production Policy Into Adapters

**Files:**
- Modify: `vllm/adapters/base.py`
- Modify: `vllm/adapters/gemma4.py`
- Modify: `vllm/adapters/qwen3_5.py`
- Modify: `vllm/model_executor/models/gemma4.py`
- Modify: `vllm/model_executor/models/qwen3_5.py`
- Modify: `tests/test_gemma4_runtime_config_policy.py`
- Modify: `tests/test_qwen35_runtime_config_policy.py`
- Modify: `tests/test_project_governance.py`

- [ ] **Step 1: Extend RuntimeModelPolicy**

In `vllm/adapters/base.py`, add normalized policy fields:

```python
@dataclass(frozen=True)
class RuntimeModelPolicy:
    force_kv_cache_dtype: str | None = None
    force_kv_cache_dtype_when: tuple[str, ...] = ()
    force_kv_cache_dtype_reason: str | None = None
    prefill_chunk_size_high_end: int | None = None
    prefill_chunk_size_standard: int | None = None
    tuning_env_overrides: dict[str, str] = field(default_factory=dict)
    model_policy: dict[str, object] = field(default_factory=dict)
    kernel_policy: dict[str, object] = field(default_factory=dict)
```

- [ ] **Step 2: Write failing adapter policy tests**

Update `tests/test_gemma4_runtime_config_policy.py` to assert stable policy:

```python
def test_gemma4_dense_adapter_provides_production_policy() -> None:
    adapter = Gemma4Adapter()
    policy = adapter.runtime_policy(_fake_dense_model_config(), _fake_runtime_config())

    assert policy.model_policy["local_decode_triton"] is True
    assert policy.model_policy["mlp_pair_fusion"] is True
    assert policy.kernel_policy["awq_decode_gemv"] is True
    assert policy.kernel_policy["awq_fused_gate_up"] is True
```

Update `tests/test_qwen35_runtime_config_policy.py` to assert:

```python
def test_qwen35_adapter_provides_stable_runtime_policy() -> None:
    adapter = Qwen35Adapter()
    policy = adapter.runtime_policy(_fake_model_config(), _fake_runtime_config())

    assert policy.model_policy["fullattn_stabilizer"] is True
    assert policy.model_policy["fullattn_use_sdpa_prefill"] is True
    assert policy.model_policy["residual_stabilizer"] is True
    assert policy.model_policy["linear_input_cap"] is True
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_gemma4_runtime_config_policy.py tests/test_qwen35_runtime_config_policy.py -q
```

Expected: FAIL until adapters expose `model_policy` and `kernel_policy`.

- [ ] **Step 4: Implement adapter-owned stable policy**

In `Gemma4Adapter.runtime_policy()`, stop reading production branch choices
from tuning env except tool-installed profile metadata. Return stable choices:

```python
return RuntimeModelPolicy(
    force_kv_cache_dtype=force_kv_dtype,
    force_kv_cache_dtype_when=("turbo_int4", "int4"),
    force_kv_cache_dtype_reason=...,
    tuning_env_overrides={
        "FASTINFERENCE_PROFILE": str(getattr(runtime_config.profile, "effective_name", "benchmark")),
    },
    model_policy={
        "local_decode_triton": True,
        "force_full_ref_attn": False,
        "legacy_fp16_ref_attn": False,
        "legacy_fullprec_kv_write": False,
        "legacy_item_path": False,
        "mlp_pair_fusion": True,
        "fp32_residual_guard_enabled": False,
        "moe_expert_cache_size": 32,
        "moe_int4_kernel_enabled": True,
        "moe_int4_kernel_strategy": "two_stage",
        "moe_prefill_grouped_enabled": False,
        "rope_cache_pool_max": 8,
    },
    kernel_policy={
        "awq_fused_scope": "all",
        "awq_fused_gemm": True,
        "awq_fused_gemm_force": False,
        "awq_decode_gemv": True,
        "awq_fused_gate_up": True,
        "awq_group32_gemv_all": not is_moe,
        "gemma4_dense_down_proj": not is_moe,
    },
)
```

In `Qwen35Adapter.runtime_policy()`, return:

```python
RuntimeModelPolicy(
    model_policy={
        "fullattn_stabilizer": True,
        "fullattn_use_sdpa_prefill": True,
        "residual_stabilizer": True,
        "linear_input_cap": True,
        "use_fla_chunk": True,
    }
)
```

- [ ] **Step 5: Install policy onto LiteInferenceConfig**

In `LiteEngine.__init__`, after `_apply_runtime_model_policy()`, copy normalized
policy into `self.inf_config` after construction:

```python
self.inf_config.model_policy = dict(self.runtime_policy.model_policy)
self.inf_config.kernel_policy = dict(self.runtime_policy.kernel_policy)
```

If `LiteInferenceConfig` is frozen or lacks attributes, add typed fields there
instead:

```python
model_policy: dict[str, object] = field(default_factory=dict)
kernel_policy: dict[str, object] = field(default_factory=dict)
```

- [ ] **Step 6: Replace production model env checks**

In `gemma4.py` and `qwen3_5.py`, replace production checks such as:

```python
_env_truthy("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN")
```

with:

```python
bool(getattr(inf_config, "model_policy", {}).get("force_full_ref_attn", False))
```

Keep diagnostic-only profiling in tool paths if needed, but ensure production
model files do not read branch decisions from env.

- [ ] **Step 7: Expand governance test**

In `tests/test_project_governance.py`, add model files to the production
allowlist guard:

```python
production_files = [
    ...
    "vllm/model_executor/models/gemma4.py",
    "vllm/model_executor/models/qwen3_5.py",
]
allowed = {"FASTINFERENCE_PROFILE"}
```

- [ ] **Step 8: Run tests**

Run:

```bash
uv run pytest tests/test_gemma4_runtime_config_policy.py tests/test_qwen35_runtime_config_policy.py tests/test_project_governance.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add vllm/adapters/base.py vllm/adapters/gemma4.py vllm/adapters/qwen3_5.py vllm/model_executor/models/gemma4.py vllm/model_executor/models/qwen3_5.py tests/test_gemma4_runtime_config_policy.py tests/test_qwen35_runtime_config_policy.py tests/test_project_governance.py
git commit -m "refactor: move model runtime policy into adapters"
```

- [ ] **Step 10: Report**

Accuracy report:
- `uv run pytest tests/test_gemma4_runtime_config_policy.py tests/test_qwen35_runtime_config_policy.py tests/test_project_governance.py -q`
- If local models exist: `SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh`

Performance report:
- If local models exist: `FASTINFERENCE_PROFILE=benchmark uv run python tests/e2e_full_benchmark.py --models qwen35_9b_awq gemma4_26b_a4b gemma4_31b_q4`
- If skipped, state missing model paths.

---

### Task 6: Move Production AWQ/Kernel Policy Out Of Env Reads

**Files:**
- Modify: `vllm/model_executor/layers/quantization/tensor.py`
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Modify: `tests/test_awq_fused_scope_policy.py`
- Modify: `tests/test_awq_fused_gemm_heuristics.py`
- Modify: `tests/test_project_governance.py`

- [ ] **Step 1: Write failing kernel policy tests**

In `tests/test_awq_fused_scope_policy.py`, add:

```python
def test_awq_policy_prefers_profile_kernel_policy_over_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "0")
    config = SimpleNamespace(
        kernel_policy={
            "awq_decode_gemv": True,
            "awq_fused_gate_up": True,
            "awq_fused_scope": "all",
        }
    )

    assert awq_decode_gemv_enabled(config) is True
```

Use the existing helper names in `tensor.py`; if helper names differ, expose a
small `config`-accepting helper and test that helper directly.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py -q
```

Expected: FAIL until production helpers accept profile kernel policy.

- [ ] **Step 3: Add config-first kernel policy helpers**

In `tensor.py`, introduce:

```python
def _kernel_policy(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    return dict(getattr(config, "kernel_policy", {}) or {})


def awq_decode_gemv_enabled(config: object | None = None) -> bool:
    policy = _kernel_policy(config)
    if "awq_decode_gemv" in policy:
        return bool(policy["awq_decode_gemv"])
    return False
```

Repeat for fused GEMM, fused gate-up, dense down projection, and group32 GEMV.
Do not let production call sites fall back to env. Keep existing env-backed
helpers only under names ending in `_tool_override` and use them only from
benchmark/tool entry points.

- [ ] **Step 4: Update AWQ fused GEMM launch policy**

In `awq_fused_gemm.py`, keep persistent profile JSON loading but stop production
manual tile overrides from env unless an explicit tool config is installed.
The production path should use:

```python
profile_entry = _lookup_persistent_blocks(...)
if profile_entry is not None:
    return profile_entry
return _select_fused_gemm_blocks(...)
```

Manual env tile overrides remain available only when
`set_awq_fused_tuning_config(..., locked=False)` is used by tools.

- [ ] **Step 5: Expand governance test**

Add kernel production files to the production guard with an allowlist for
documented tool-only functions. If the file still contains `FASTINFERENCE_AWQ_*`,
the test must assert those names appear only in comments, tests, or tool-override
helpers.

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py tests/test_project_governance.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/model_executor/layers/quantization/tensor.py vllm/kernels/triton/awq_fused_gemm.py tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py tests/test_project_governance.py
git commit -m "refactor: resolve production awq policy from profiles"
```

- [ ] **Step 8: Report**

Accuracy report:
- `uv run pytest tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py tests/test_project_governance.py -q`
- `uv run pytest tests/test_awq_fused_gemm_numerics.py -q`

Performance report:
- If GPU is available: run existing AWQ microbenchmark for the validated shapes.
- If skipped, state GPU/model availability.

---

### Task 7: Update Documentation And Promotion Guardrails

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE_LITE.md`
- Modify: `docs/CAPABILITY_MATRIX.md`
- Modify: `docs/superpowers/specs/2026-05-22-runtime-profile-convergence-design.md` if implementation reveals a necessary correction.
- Modify: `tests/test_project_governance.py`

- [ ] **Step 1: Update README environment section**

Replace the current environment table with:

```markdown
## Runtime Profiles

Production inference is selected with `FASTINFERENCE_PROFILE`.

| Profile | Purpose |
| :--- | :--- |
| `auto` | Resolve the recommended stable profile from model and device capabilities. |
| `benchmark` | Reproduce repository benchmark defaults. |
| `latency` | Prefer low TPOT and single-request serving. |
| `throughput` | Prefer aggregate TPS for supported batch shapes. |
| `accuracy` | Prefer conservative branches when accuracy is the priority. |

Benchmark and diagnostic tools may expose additional tuning switches. Those
switches are not production runtime controls; validated results must be promoted
into a runtime profile.
```

- [ ] **Step 2: Update architecture docs**

In `docs/ARCHITECTURE_LITE.md`, add:

```markdown
Runtime configuration now flows through `RuntimeProfileRegistry`:

LLM / AsyncLLM / Server
  -> config_builder
  -> RuntimeProfileRegistry
  -> RuntimeConfig
  -> LiteEngine

Production code treats profile data as the strategy source of truth. Benchmark
tools remain the place for broad tuning exploration.
```

- [ ] **Step 3: Update capability matrix**

Change experimental Gemma4 env references to profile/tool language:

```markdown
Gemma4-26B MoE int4 decode kernel is supported through the benchmark and
latency profiles. Alternative strategies remain benchmark-tool experiments and
are not production runtime switches.
```

- [ ] **Step 4: Add docs governance test**

In `tests/test_project_governance.py`, add:

```python
def test_readme_documents_profiles_instead_of_runtime_env_matrix() -> None:
    readme = _read("README.md")
    assert "FASTINFERENCE_PROFILE" in readme
    assert "Runtime Profiles" in readme
    assert "FASTINFERENCE_KV_TYPE" not in readme
    assert "FASTINFERENCE_FUSION_LEVEL" not in readme
```

- [ ] **Step 5: Run docs/governance tests**

Run: `uv run pytest tests/test_project_governance.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/ARCHITECTURE_LITE.md docs/CAPABILITY_MATRIX.md tests/test_project_governance.py
git commit -m "docs: document runtime profiles"
```

- [ ] **Step 7: Report**

Accuracy report:
- `uv run pytest tests/test_project_governance.py -q`

Performance report:
- Skipped: docs and guardrails only.

---

### Task 8: Final Regression And Baseline Performance Report

**Files:**
- No required source edits unless regressions are found.
- Optional: update `docs/GEMMA4_31B_PERF_WRAPUP.md` or add a new dated report if the user wants persisted benchmark results.

- [ ] **Step 1: Run fast regression**

Run:

```bash
bash tests/run_regression_suite.sh
```

Expected: PASS.

- [ ] **Step 2: Run targeted profile tests**

Run:

```bash
uv run pytest tests/test_runtime_profile.py tests/test_kv_default_policy.py tests/test_runtime_stats_export.py tests/test_project_governance.py -q
```

Expected: PASS.

- [ ] **Step 3: Run inference accuracy regression**

Run:

```bash
SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh
```

Expected: PASS for available models. If models are unavailable, record exact
missing paths from script output.

- [ ] **Step 4: Run benchmark profile report**

Run:

```bash
FASTINFERENCE_PROFILE=benchmark uv run python tests/e2e_full_benchmark.py
```

Expected: report includes profile metadata and phase metrics.

- [ ] **Step 5: Capture final summary**

Final implementation response must include:

```text
Accuracy report:
- bash tests/run_regression_suite.sh: PASS/FAIL
- uv run pytest ...: PASS/FAIL
- SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh: PASS/FAIL/SKIPPED

Performance report:
- FASTINFERENCE_PROFILE=benchmark uv run python tests/e2e_full_benchmark.py: PASS/FAIL/SKIPPED
- TinyLlama metrics:
- Qwen3.5 metrics:
- Gemma4-26B metrics:
- Gemma4-31B metrics:
- Baseline deltas:
```

- [ ] **Step 6: Commit any final report doc if created**

If a persistent benchmark report is added:

```bash
git add docs/<report-file>.md
git commit -m "docs: add runtime profile benchmark report"
```
