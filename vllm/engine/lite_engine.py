# SPDX-License-Identifier: Apache-2.0
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import torch

from vllm.adapters import get_model_adapter
from vllm.config import VllmConfig
from vllm.engine.errors import (
    BackgroundLoopError,
    EngineFatalError,
    InvalidRequestError,
    RequestRejectedError,
)
from vllm.engine.inference_config import LiteInferenceConfig
from vllm.engine.initialization import (
    BlockAllocator,
    FlatKVCacheAllocator,
    LiteRuntimeAssembler,
    MemoryAuditor,
)
from vllm.engine.initialization.kv_cache_allocator import (
    compute_kv_scale_theory_bytes,
    compute_kv_theory_bytes,
    resolve_layer_kv_specs,
)
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb, select_loadtime_policy
from vllm.engine.lora_runtime import LoRARuntimeRegistry
from vllm.engine.model_surface import resolve_model_surface
from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.request_builder import LiteRequestBuilder
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.engine.runtime_config import RuntimeConfig
from vllm.engine.runtime_observer import NullRuntimeObserver
from vllm.engine.runtime_planner import RuntimePlanner
from vllm.engine.sampling_driver import SamplingDriver
from vllm.logger import init_logger
from vllm.lora.manager import LoRAManager
from vllm.model_executor.model_loader import get_model
from vllm.outputs import RequestOutput
from vllm.policies import build_generation_policies
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def expand_metadata_for_paged_attention(
    bs: int,
    seq: int,
    is_prefill: bool,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    q_device: torch.device,
    seq_lens_cpu: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expands seq_lens and block_tables for PagedAttention kernels during prefill.
    Standardizes 'Chunked Prefill' logic across Llama and Qwen architectures.

    When ``seq_lens_cpu`` is provided (a Python list of ints), this function
    avoids any GPU->CPU sync via ``.item()``; the caller is expected to have
    already surfaced per-request lengths to the host side.
    """
    if seq > 1 and is_prefill:
        if bs == 1:
            if seq_lens_cpu is not None and len(seq_lens_cpu) >= 1:
                end_pos = int(seq_lens_cpu[0])
            else:
                end_pos = int(seq_lens[0].item())
            start_pos = end_pos - seq
            seq_lens_ext = torch.arange(
                start_pos + 1, end_pos + 1, device=q_device, dtype=torch.int32
            )
            block_tables_ext = block_tables.expand(seq, -1).contiguous()
        else:
            # Batched chunked prefill: flatten tokens in batch-major order.
            seq_lens_ext_parts = []
            block_tables_ext_parts = []
            for bi in range(bs):
                if seq_lens_cpu is not None and len(seq_lens_cpu) > bi:
                    end_pos_b = int(seq_lens_cpu[bi])
                else:
                    end_pos_b = int(seq_lens[bi].item())
                start_pos_b = end_pos_b - seq
                seq_lens_ext_parts.append(
                    torch.arange(
                        start_pos_b + 1,
                        end_pos_b + 1,
                        device=q_device,
                        dtype=torch.int32,
                    )
                )
                block_tables_ext_parts.append(block_tables[bi : bi + 1].expand(seq, -1))
            seq_lens_ext = torch.cat(seq_lens_ext_parts, dim=0)
            block_tables_ext = torch.cat(block_tables_ext_parts, dim=0).contiguous()
        return seq_lens_ext, block_tables_ext

    return seq_lens, block_tables


class LiteEngine:
    def __init__(self, vllm_config: VllmConfig):
        self._initialize_before_model_load(vllm_config)
        self._load_model_and_detect_capabilities()

        # Pre-allocate Block-based KV Cache (paged: block table + fixed pool).
        self.inf_config = LiteInferenceConfig(
            kv_type=self.runtime_config.kv_cache_dtype,
            k_scale=self.runtime_config.k_scale,
            v_scale=self.runtime_config.v_scale,
            fusion_level=self.runtime_config.fusion_level,
            block_size=self.runtime_config.block_size,
            max_model_len=self.runtime_config.kv_max_model_len,
            max_active_requests=self.runtime_config.kv_max_active_requests,
            use_prompt_guard=self.runtime_config.use_prompt_guard,
            paged_attn_num_warps=self.runtime_config.paged_attn_num_warps,
            paged_attn_num_stages=self.runtime_config.paged_attn_num_stages,
            paged_attn_num_warps_global=self.runtime_config.paged_attn_num_warps_global,
            paged_attn_num_stages_global=self.runtime_config.paged_attn_num_stages_global,
            paged_attn_num_warps_local=self.runtime_config.paged_attn_num_warps_local,
            paged_attn_num_stages_local=self.runtime_config.paged_attn_num_stages_local,
            gemma4_c1_preset=(
                self.runtime_config.gemma4_c1_preset
                or bool(self.runtime_policy.model_policy.get("gemma4_c1_preset"))
            ),
            tuning_env=self._active_tuning_env,
            model_policy=dict(self.runtime_policy.model_policy),
            kernel_policy=dict(self.runtime_policy.kernel_policy),
            kv_select_ratio=self.runtime_config.kv_select_ratio,
            kv_select_min_blocks=self.runtime_config.kv_select_min_blocks,
        )

        self._initialize_loaded_runtime()

    def _initialize_before_model_load(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._async_engine_lock: Any | None = None
        self._fatal_error: BackgroundLoopError | None = None
        self.model_config = vllm_config.model_config
        self.device = torch.device("cuda:0")
        self.runtime_config: RuntimeConfig = cast(
            RuntimeConfig,
            getattr(vllm_config, "runtime_config", None)
            or RuntimeConfig.from_vllm_config(vllm_config),
        )
        object.__setattr__(self.vllm_config, "runtime_config", self.runtime_config)
        self._requested_policy_mode = self.runtime_config.policy_mode
        self.observer = (
            getattr(vllm_config, "runtime_observer", None) or NullRuntimeObserver()
        )
        self.adapter = get_model_adapter(None, self.model_config)
        self.runtime_policy = self.adapter.runtime_policy(
            self.model_config,
            self.runtime_config,
        )
        self._apply_runtime_model_policy()
        self._install_runtime_policy_on_runtime_config(self.runtime_policy)
        object.__setattr__(self.vllm_config, "runtime_config", self.runtime_config)
        self._install_tuning_configs_for_model(self.runtime_policy)

    def _load_model_and_detect_capabilities(self) -> None:
        print(f">>> LiteEngine: Loading {self.model_config.model}...")
        self.model = get_model(vllm_config=self.vllm_config)
        print(f">>> LiteEngine: Model Type: {type(self.model)}")
        self.tokenizer = None
        self.adapter = get_model_adapter(self.model, self.model_config)
        self.execution_policy = select_loadtime_policy(
            model_config=self.model_config,
            quant_config=getattr(self.vllm_config, "quant_config", None),
            policy_mode=self._requested_policy_mode,  # type: ignore[arg-type]
        )
        self.model_capabilities = self.adapter.detect(self.model, self.model_config)
        self.model_surface = resolve_model_surface(
            model_name=str(getattr(self.model_config, "model", "")),
            capabilities=self.model_capabilities,
        )
        object.__setattr__(
            self.vllm_config, "model_capabilities", self.model_capabilities
        )
        self.num_attention_heads = self.model_capabilities.num_attention_heads
        self.num_kv_heads = self.model_capabilities.num_kv_heads
        self.head_size = self.model_capabilities.head_dim
        self.num_layers = self.model_capabilities.num_layers
        print(
            f">>> LiteEngine: Verified Dimensions: {self.num_attention_heads} Q-heads, "
            f"{self.num_kv_heads} KV-heads, {self.head_size} head_dim"
        )
        self._layer_kv_specs = resolve_layer_kv_specs(self.model, self.num_layers)
        fused_stage = next(
            (
                value
                for key, value in self.runtime_policy.tuning_env_overrides.items()
                if key.endswith("AWQ_FUSED_SCOPE")
            ),
            None,
        )
        if fused_stage is not None:
            print(
                ">>>> LiteEngine: fused rollout stage="
                f"{fused_stage} "
                "(from runtime profile/model policy)."
            )

    def _initialize_loaded_runtime(self) -> None:
        planner = RuntimePlanner(
            self.runtime_config,
            self.model_capabilities,
            self.runtime_policy,
        )
        execution_plan = planner.build_execution_plan(
            self.execution_policy.max_active_requests
        )
        kv_plan = planner.build_kv_cache_plan(execution_plan)
        gpu_total_gb = get_total_gpu_memory_gb()
        is_high_end_gpu = execution_plan.is_high_end_gpu
        if is_high_end_gpu:
            print(
                f">>>> LiteEngine: High-end GPU detected ({gpu_total_gb:.1f}GB). "
                "Enabling aggressive optimization."
            )

        self.block_size = execution_plan.block_size
        self.max_model_len = execution_plan.max_model_len
        self.max_active_requests = execution_plan.max_active_requests
        self.num_blocks_per_seq = execution_plan.num_blocks_per_seq
        self.num_total_blocks = execution_plan.num_total_blocks
        self._step_token_budget = execution_plan.step_token_budget
        self._prefill_chunk_size = execution_plan.prefill_chunk_size
        self._decode_priority_enabled = execution_plan.decode_priority_enabled
        self._prefill_reserved_tokens = execution_plan.prefill_reserved_tokens
        self._prefill_reserve_backlog = execution_plan.prefill_reserve_backlog
        self._prefill_catchup_ratio = execution_plan.prefill_catchup_ratio
        self._prefill_microbatch_size = execution_plan.prefill_microbatch_size
        self._apply_adapter_admission_cap()

        print(
            ">>>> LiteEngine: Step scheduler "
            f"(token_budget={self._step_token_budget}, "
            f"decode_priority={self._decode_priority_enabled}, "
            f"prefill_reserved_tokens={self._prefill_reserved_tokens}, "
            f"prefill_reserve_backlog={self._prefill_reserve_backlog}, "
            f"prefill_catchup_ratio={self._prefill_catchup_ratio:.2f}, "
            f"prefill_microbatch={self._prefill_microbatch_size})"
        )

        custom_runtime_components = self.adapter.build_executors(
            model=self.model,
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            observer=self.observer,
            device=self.device,
            max_active_requests=self.max_active_requests,
        )
        if custom_runtime_components is not None:
            self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.kv_scale_caches: list[
                tuple[torch.Tensor | None, torch.Tensor | None]
            ] = []
            block_allocator = None
            self.block_size = custom_runtime_components.kv_block_manager.block_size
            self.num_blocks_per_seq = (
                custom_runtime_components.kv_block_manager.num_blocks_per_seq
            )
            print(">>>> LiteEngine: Using model-owned KV runtime components.")
        else:
            self._resolve_kv_metadata(kv_plan)
            kv_theory_bytes = compute_kv_theory_bytes(
                layer_kv_specs=self._layer_kv_specs,
                num_layers=self.num_layers,
                num_total_blocks=self.num_total_blocks,
                block_size=self.block_size,
                fallback_num_kv_heads=self.num_kv_heads,
                fallback_kv_head_dim=self.kv_head_dim,
                kv_dtype=self.kv_dtype,
                needs_scale_cache=kv_plan.needs_scale_cache,
            )
            print(
                f">>>> LiteEngine: Allocating KV Cache on {self.device} "
                f"({self.max_active_requests} seq slots, "
                f"{self.max_model_len} tokens/seq cap, {self.num_layers} layers, "
                f"block={self.block_size} tok, dtype={self.kv_dtype}, "
                f"~{kv_theory_bytes / (1024**3):.3f} GiB theoretical)"
            )
            mem_before_kv = int(torch.cuda.memory_allocated(self.device))
            allocator = FlatKVCacheAllocator(
                num_layers=self.num_layers,
                num_total_blocks=self.num_total_blocks,
                block_size=self.block_size,
                device=self.device,
            )
            self.kv_caches, self.kv_scale_caches, _ = allocator.allocate(
                layer_kv_specs=self._layer_kv_specs,
                kv_dtype=self.kv_dtype,
                kv_head_dim=self.kv_head_dim,
                fallback_num_kv_heads=self.num_kv_heads,
                fallback_kv_head_dim=self.kv_head_dim,
                needs_scale_cache=kv_plan.needs_scale_cache,
            )
            block_allocator = BlockAllocator(num_total_blocks=self.num_total_blocks)
            print(">>>> LiteEngine: KV Cache allocated successfully.")
            mem_after_kv = int(torch.cuda.memory_allocated(self.device))
            auditor = MemoryAuditor(
                device=self.device,
                topn=self.runtime_config.memory_audit_topn,
            )
            audit = auditor.audit(self.model)
            self._print_startup_memory_audit(
                kv_plan=kv_plan,
                kv_theory_bytes=kv_theory_bytes,
                mem_before_kv=mem_before_kv,
                mem_after_kv=mem_after_kv,
                audit=audit,
            )

        # slot_mapping maps batch tokens to physical indices
        self.scheduler = RequestScheduler(
            self.max_active_requests, runtime_config=self.runtime_config
        )
        self.lora_registry = LoRARuntimeRegistry()
        self.lora_manager = LoRAManager(self.model)
        self.lora_manager.bind_to_model(self.model)
        self.policies: Any | None = None
        self.sampling_driver: SamplingDriver | None = None
        self.output_pipeline: OutputPipeline | None = None
        self.request_builder: LiteRequestBuilder | None = None
        self._record_model_surface_event()
        self._queue_timeout_s = float(self.runtime_config.queue_timeout_s)

        # Pre-allocate tensors for SYNC FAST PATH (BS=1 to max_active_requests)
        # These will be reused to avoid Python object creation in every decode step.
        self._fast_input_ids = torch.empty(
            (self.max_active_requests, 1), dtype=torch.long, device=self.device
        )
        self._fast_positions = torch.empty(
            (self.max_active_requests, 1), dtype=torch.long, device=self.device
        )
        self._fast_slot_mapping = torch.empty(
            (self.max_active_requests,), dtype=torch.long, device=self.device
        )
        self._fast_seq_lens = torch.empty(
            (self.max_active_requests,), dtype=torch.int32, device=self.device
        )

        supports_chunked_prefill = bool(
            getattr(self.model_capabilities, "supports_chunked_prefill", True)
        )
        effective_prefill_chunk_size = (
            self._prefill_chunk_size if supports_chunked_prefill else self.max_model_len
        )
        effective_prefill_microbatch_size = (
            self._prefill_microbatch_size if supports_chunked_prefill else 1
        )
        effective_max_prefill_chunk_size = (
            self.runtime_config.max_prefill_chunk_size
            if supports_chunked_prefill
            else self.max_model_len
        )

        runtime_components = LiteRuntimeAssembler.assemble(
            block_allocator=block_allocator,
            kv_caches=self.kv_caches,
            kv_scale_caches=self.kv_scale_caches,
            num_blocks_per_seq=self.num_blocks_per_seq,
            block_size=self.block_size,
            device=self.device,
            max_model_len=self.max_model_len,
            num_layers=self.num_layers,
            inf_config=self.inf_config,
            stack_per_layer_carries=self._stack_per_layer_carries,
            split_per_layer_carries=self._split_per_layer_carries,
            model=self.model,
            fast_input_ids=self._fast_input_ids,
            fast_positions=self._fast_positions,
            fast_slot_mapping=self._fast_slot_mapping,
            fast_seq_lens=self._fast_seq_lens,
            step_token_budget=self._step_token_budget,
            decode_priority_enabled=self._decode_priority_enabled,
            prefill_chunk_size=effective_prefill_chunk_size,
            prefill_reserved_tokens=self._prefill_reserved_tokens,
            prefill_reserve_backlog=self._prefill_reserve_backlog,
            prefill_catchup_ratio=self._prefill_catchup_ratio,
            prefill_microbatch_size=effective_prefill_microbatch_size,
            min_prefill_chunk_size=self.runtime_config.min_prefill_chunk_size,
            max_prefill_chunk_size=effective_max_prefill_chunk_size,
            prefill_sla_ttft_ms=self.runtime_config.prefill_sla_ttft_ms,
            max_active_requests=self.max_active_requests,
            scheduler_policy=self.runtime_config.scheduler_policy,
            backend_policy=self.runtime_config.backend_policy,
            scheduler=self.scheduler,
            observer=self.observer,
            lora_registry=self.lora_registry,
            queue_timeout_s=self._queue_timeout_s,
            custom_runtime_components=custom_runtime_components,
        )
        self.kv_block_manager = runtime_components.kv_block_manager
        self.input_batch_builder = runtime_components.input_batch_builder
        self.multimodal_processor = runtime_components.multimodal_processor
        self.prefill_executor = runtime_components.prefill_executor
        self.decode_executor = runtime_components.decode_executor
        self.step_scheduler = runtime_components.step_scheduler
        self.step_scheduler.set_verified_decode_batch_sizes(
            self.runtime_policy.verified_decode_batch_sizes
        )
        self.execution_backend = runtime_components.execution_backend
        self.runtime_controller = runtime_components.runtime_controller

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def _apply_adapter_admission_cap(self) -> None:
        admission_cap = getattr(self.adapter, "admission_cap", None)
        if not callable(admission_cap):
            return
        capped = int(
            admission_cap(
                current_max_active_requests=self.max_active_requests,
                max_model_len=self.max_model_len,
                runtime_config=self.runtime_config,
            )
        )
        if capped < self.max_active_requests:
            self.max_active_requests = capped
            self.num_total_blocks = self.num_blocks_per_seq * capped
            object.__setattr__(self.runtime_config, "kv_max_active_requests", capped)

    def _apply_runtime_model_policy(self) -> None:
        force_kv_dtype = self.runtime_policy.force_kv_cache_dtype
        if not force_kv_dtype:
            return
        current = str(self.runtime_config.kv_cache_dtype).lower()
        allowed_current = self.runtime_policy.force_kv_cache_dtype_when
        if allowed_current and current not in allowed_current:
            return
        reason = self.runtime_policy.force_kv_cache_dtype_reason
        if reason:
            print(f">>>> LiteEngine: {reason}")
        self.runtime_config = replace(
            self.runtime_config,
            kv_cache_dtype=force_kv_dtype,
        )

    def _install_runtime_policy_on_runtime_config(self, runtime_policy: Any) -> None:
        object.__setattr__(
            self.runtime_config,
            "model_policy",
            dict(getattr(runtime_policy, "model_policy", {}) or {}),
        )
        object.__setattr__(
            self.runtime_config,
            "kernel_policy",
            dict(getattr(runtime_policy, "kernel_policy", {}) or {}),
        )

    def _record_model_surface_event(self) -> None:
        surface = self.model_surface
        self.observer.on_model_surface_resolved(
            event_name=surface.event_name,
            model_name=surface.model_name,
            model_type=surface.model_type,
            status=surface.status,
            reason=surface.reason,
        )
        log = logger.warning if surface.status == "experimental" else logger.info
        log(
            "LiteEngine model surface event=%s model=%s model_type=%s "
            "status=%s reason=%s",
            surface.event_name,
            surface.model_name,
            surface.model_type,
            surface.status,
            surface.reason,
        )

    def _install_tuning_configs_for_model(self, runtime_policy: Any) -> None:
        tuning_env: dict[str, str] = dict(self.runtime_config.tuning_env or {})
        for key, value in runtime_policy.tuning_env_overrides.items():
            tuning_env.setdefault(str(key), str(value))

        self._active_tuning_env = tuning_env

        try:
            from vllm.kernels.triton.awq_fused_gemm import (
                set_awq_fused_tuning_config,
            )

            set_awq_fused_tuning_config(dict(tuning_env), locked=True)
        except Exception:
            logger.debug("Unable to install AWQ fused tuning config", exc_info=True)
        try:
            from vllm.model_executor.layers.quantization.tensor import (
                set_awq_tensor_tuning_config,
            )

            set_awq_tensor_tuning_config(dict(tuning_env), locked=True)
        except Exception:
            logger.debug("Unable to install AWQ tensor tuning config", exc_info=True)
        try:
            self.adapter.install_tuning_config(tuning_env)
        except Exception:
            logger.debug("Unable to install model tuning config", exc_info=True)

    def _resolve_kv_metadata(self, kv_plan: Any) -> None:
        """Derive KV dtype/head_dim from the planner and update inf_config."""
        if kv_plan.kv_dtype == torch.uint8:
            self.inf_config.kv_type = "turbo_int4"
            print(
                ">>>> LiteEngine: KV Cache quantized to TurboQuant INT4 "
                "(uint8 packed) [NEW]"
            )
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
            print(
                f">>>> LiteEngine: Using KV scales: K={self.inf_config.k_scale}, "
                f"V={self.inf_config.v_scale}"
            )
        elif kv_plan.kv_dtype == torch.float8_e4m3fn:
            self.inf_config.kv_type = "fp8"
            print(">>>> LiteEngine: KV Cache quantized to FP8 (e4m3fn) [STABLE]")
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
        else:
            self.inf_config.kv_type = "fp16"
            print(
                ">>>> LiteEngine: KV Cache using full precision (BF16/FP16) [ACCURATE]"
            )
            if self.model_capabilities.preferred_kv_dtype == "bfloat16":
                print(">>>> LiteEngine: KV Cache dtype bfloat16 (Qwen3.5)")
            else:
                print(">>>> LiteEngine: KV Cache dtype float16")
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim

    def _print_startup_memory_audit(
        self,
        *,
        kv_plan: Any,
        kv_theory_bytes: int,
        mem_before_kv: int,
        mem_after_kv: int,
        audit: dict[str, Any],
    ) -> None:
        """Print GPU memory breakdown and tensor footprint after KV allocation."""
        kv_delta_bytes = mem_after_kv - mem_before_kv
        total_gb = _bytes_to_gib(mem_after_kv)
        weights_gb = _bytes_to_gib(mem_before_kv)
        kv_delta_gb = _bytes_to_gib(kv_delta_bytes)
        gpu_total_gb = get_total_gpu_memory_gb()
        params_total_bytes = int(audit["params_total_bytes"])
        buffers_total_bytes = int(audit["buffers_total_bytes"])
        awq_cache_bytes = int(audit["awq_cache_bytes"])
        accounted_before_kv = params_total_bytes + buffers_total_bytes + awq_cache_bytes
        other_cuda_overhead = max(0, mem_before_kv - accounted_before_kv)
        kv_data_theory = compute_kv_theory_bytes(
            layer_kv_specs=self._layer_kv_specs,
            num_layers=self.num_layers,
            num_total_blocks=self.num_total_blocks,
            block_size=self.block_size,
            fallback_num_kv_heads=self.num_kv_heads,
            fallback_kv_head_dim=self.kv_head_dim,
            kv_dtype=self.kv_dtype,
            needs_scale_cache=False,
        )
        kv_scale_theory = (
            compute_kv_scale_theory_bytes(
                layer_kv_specs=self._layer_kv_specs,
                num_layers=self.num_layers,
                num_total_blocks=self.num_total_blocks,
                block_size=self.block_size,
                fallback_num_kv_heads=self.num_kv_heads,
                fallback_kv_head_dim=self.kv_head_dim,
            )
            if kv_plan.needs_scale_cache
            else 0
        )
        print(
            ">>>> LiteEngine: GPU memory breakdown (torch.cuda.memory_allocated; "
            "host RSS not included — large GGUF load is often CPU anon-rss):"
        )
        print(f"     before_KV (weights + overhead): {weights_gb:.3f} GiB")
        print(
            f"     KV pool (delta alloc):          {kv_delta_gb:.3f} GiB  "
            f"(theory {kv_theory_bytes / (1024**3):.3f} GiB)"
        )
        print(
            f"     after_KV total:                 {total_gb:.3f} GiB  /  "
            f"GPU cap ~{gpu_total_gb:.1f} GiB"
        )
        print(">>>> LiteEngine: Startup memory audit (CUDA tensors):")
        print(
            "     model params:                  "
            f"{_bytes_to_gib(params_total_bytes):.3f} GiB  "
            f"({audit['params_count']} tensors)"
        )
        for dtype_name, nbytes in sorted(
            audit["params_dtype_bytes"].items(),
            key=lambda kv: -int(kv[1]),
        ):
            print(
                f"       - params[{dtype_name:<10}]         "
                f"{_bytes_to_gib(int(nbytes)):.3f} GiB"
            )
        if audit["params_top"]:
            print(f"     top params by size (Top-{audit['topn']}):")
            for row in audit["params_top"]:
                print(
                    "       - "
                    f"{row['name']} "
                    f"shape={tuple(row['shape'])} dtype={row['dtype']} "
                    f"size={_bytes_to_gib(int(row['bytes'])):.3f} GiB"
                )
        print(
            "     model buffers:                 "
            f"{_bytes_to_gib(buffers_total_bytes):.3f} GiB  "
            f"({audit['buffers_count']} tensors)"
        )
        for dtype_name, nbytes in sorted(
            audit["buffers_dtype_bytes"].items(),
            key=lambda kv: -int(kv[1]),
        ):
            print(
                f"       - buffers[{dtype_name:<9}]        "
                f"{_bytes_to_gib(int(nbytes)):.3f} GiB"
            )
        if audit["buffers_top"]:
            print(f"     top buffers by size (Top-{audit['topn']}):")
            for row in audit["buffers_top"]:
                print(
                    "       - "
                    f"{row['name']} "
                    f"shape={tuple(row['shape'])} dtype={row['dtype']} "
                    f"size={_bytes_to_gib(int(row['bytes'])):.3f} GiB"
                )
        print(
            "     AWQ dense cache (global):      "
            f"{_bytes_to_gib(awq_cache_bytes):.3f} GiB"
        )
        print(
            "     other CUDA overhead (est.):    "
            f"{_bytes_to_gib(other_cuda_overhead):.3f} GiB"
        )
        print(
            "     KV theory split:               "
            f"data={_bytes_to_gib(kv_data_theory):.3f} GiB, "
            f"scales={_bytes_to_gib(kv_scale_theory):.3f} GiB"
        )
        if gpu_total_gb > 0 and total_gb > 0.85 * gpu_total_gb:
            print(
                "     [Warn] Total allocated is high vs GPU size; "
                "reduce the selected runtime profile context/concurrency limits "
                "or choose the accuracy profile."
            )

    @property
    def active_request_count(self) -> int:
        """Number of in-flight requests (for debugging / test harness guards)."""
        return self.scheduler.active_request_count

    @staticmethod
    def _stack_per_layer_carries(
        req_dicts: list[RequestState], num_layers: int, key: str
    ) -> list[torch.Tensor | None]:
        """Batch (B, ...) tensors per layer for Qwen3.5 linear-attn streaming state."""
        stacked: list[torch.Tensor | None] = []
        for li in range(num_layers):
            parts = [getattr(r, key)[li] for r in req_dicts]
            if all(p is None for p in parts):
                stacked.append(None)
            else:
                if any(p is None for p in parts):
                    raise RuntimeError(
                        f"Mixed None/non-None in batched decode for {key}[layer={li}]"
                    )
                # Each request stores (1, ...) slices; concatenate batch dim,
                # do not stack
                # (stack would produce (B, 1, ...) and break Qwen3.5 conv carry cat).
                stacked.append(torch.cat(parts, dim=0))
        return stacked

    @staticmethod
    def _split_per_layer_carries(
        stacked: list[torch.Tensor | None],
        req_dicts: list[RequestState],
        key: str,
    ) -> None:
        for li, t in enumerate(stacked):
            for i, r in enumerate(req_dicts):
                carry = getattr(r, key)
                if t is None:
                    carry[li] = None
                else:
                    carry[li] = t[i : i + 1].contiguous()

    def register_lora_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str | None = None,
        lora_int_id: int | None = None,
    ) -> dict[str, Any]:
        if not bool(getattr(self.model_capabilities, "supports_lora", False)):
            raise ValueError("model does not support LoRA")
        request = self.lora_registry.register_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
            lora_int_id=lora_int_id,
        )
        self.lora_manager.register_adapter(
            lora_name=request.lora_name,
            lora_path=request.lora_path,
        )
        return {
            "lora_name": request.lora_name,
            "lora_int_id": request.lora_int_id,
            "lora_path": request.lora_path,
        }

    def unregister_lora_adapter(self, lora_name: str) -> bool:
        return self.lora_registry.unregister_adapter(lora_name)

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_id: str | None = None,
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ):
        fatal_error = getattr(self, "_fatal_error", None)
        if fatal_error is not None:
            raise EngineFatalError(f"engine is fatal: {fatal_error}")
        if self.policies is None:
            self.policies = build_generation_policies(
                str(self.model_config.model), self.tokenizer, self.adapter
            )
            self.sampling_driver = SamplingDriver(
                self.tokenizer,
                getattr(self.model_config, "hf_config", None),
                self.policies,
                use_legacy=self.runtime_config.use_legacy_sampling,
            )
            self.output_pipeline = OutputPipeline(
                self.tokenizer,
                self.policies,
                self.sampling_driver,
                max_model_len=self.max_model_len,
            )
            self.request_builder = LiteRequestBuilder(
                tokenizer=self.tokenizer,
                policies=self.policies,
                device=self.device,
                num_layers=self.num_layers,
                max_model_len=self.max_model_len,
                max_tokens_cap=self.execution_policy.max_tokens_cap,
                default_min_new_tokens=self.runtime_config.default_min_new_tokens,
                multimodal_processor=self.multimodal_processor,
            )
            self.execution_backend.sampling_driver = self.sampling_driver
            self.execution_backend.output_coordinator = self.output_pipeline
        if not self.scheduler.has_queue_capacity():
            reason = (
                "request queue full "
                f"(running={self.scheduler.running_request_count}, "
                f"queued={self.scheduler.queued_request_count})"
            )
            self.observer.on_request_rejected(request_id, reason)
            raise RequestRejectedError(reason)

        validate_request = getattr(self.adapter, "validate_request", None)
        if validate_request is not None:
            try:
                validate_request(
                    sampling_params=sampling_params,
                    lora_id=lora_id,
                    lora_request=lora_request,
                    multi_modal_data=multi_modal_data,
                )
            except ValueError as exc:
                reason = str(exc)
                self.observer.on_request_rejected(request_id, reason)
                raise InvalidRequestError(reason) from exc

        try:
            resolved_lora = self.lora_registry.resolve_adapter(
                lora_id=lora_id,
                lora_request=lora_request,
            )
            if resolved_lora is not None and not bool(
                getattr(self.model_capabilities, "supports_lora", False)
            ):
                raise ValueError("model does not support LoRA")
            if multi_modal_data is not None and not bool(
                getattr(self.model_capabilities, "supports_multimodal", False)
            ):
                raise ValueError("model does not support multimodal input")
            if resolved_lora is not None and not self.lora_manager.has_adapter(
                resolved_lora.lora_name
            ):
                self.lora_manager.register_adapter(
                    lora_name=resolved_lora.lora_name,
                    lora_path=resolved_lora.lora_path,
                )
            if self.request_builder is None:
                raise RuntimeError("request builder was not initialized")
            request_state = self.request_builder.build(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                lora_id=resolved_lora.lora_name if resolved_lora is not None else None,
                lora_int_id=resolved_lora.lora_int_id
                if resolved_lora is not None
                else None,
                lora_path=resolved_lora.lora_path
                if resolved_lora is not None
                else None,
                multi_modal_data=multi_modal_data,
            )
            if not bool(
                getattr(self.model_capabilities, "supports_chunked_prefill", True)
            ):
                max_prefill_tokens = min(self.max_model_len, self._step_token_budget)
                if len(request_state.input_ids) > max_prefill_tokens:
                    raise ValueError(
                        "request prompt too long for full-prefill model "
                        f"({len(request_state.input_ids)} > {max_prefill_tokens})"
                    )
            self.multimodal_processor.prepare_request(request_state)
        except ValueError as exc:
            reason = str(exc)
            self.observer.on_request_rejected(request_id, reason)
            raise InvalidRequestError(reason) from exc
        if (
            not bool(getattr(self.model_capabilities, "supports_chunked_prefill", True))
            and self.scheduler.active_request_count >= self.max_active_requests
        ):
            reason = (
                "request admission cap reached "
                f"({self.scheduler.active_request_count}/{self.max_active_requests})"
            )
            self.observer.on_request_rejected(request_id, reason)
            raise RequestRejectedError(reason)

        self.execution_backend.maybe_apply_prefix_cache(request_state)
        self.scheduler.enqueue_request(request_id, request_state)
        self.lora_registry.on_request_added(request_state.lora_id)
        self.observer.on_request_added(request_id, request_state)

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        async for output in self.scheduler.get_request_stream(request_id):
            yield output

    def abort_request(self, request_id: str) -> None:
        try:
            req = self.scheduler.get_request(request_id)
        except KeyError:
            return
        if self.output_pipeline is None:
            raise RuntimeError("output pipeline was not initialized")
        output = self.output_pipeline.build_abort_output(request_id, req)
        self.scheduler.publish_output(request_id, output)
        self.execution_backend.release_request(request_id)
        self.observer.on_request_aborted(request_id)

    def close_request_stream(self, request_id: str) -> None:
        self.scheduler.close_request_stream(request_id)

    def handle_background_error(self, exc: BaseException) -> None:
        if self._fatal_error is not None:
            return
        self._fatal_error = (
            exc
            if isinstance(exc, BackgroundLoopError)
            else BackgroundLoopError(str(exc))
        )
        request_ids = self.scheduler.request_ids()
        self.observer.on_background_error(self._fatal_error, request_ids)
        for request_id in request_ids:
            self.scheduler.publish_exception(request_id, self._fatal_error)
        for request_id in request_ids:
            try:
                self.execution_backend.release_request(request_id)
            except Exception:
                # The engine is already fatal; continue teardown so every
                # stream has a terminal notification and every sibling gets a
                # chance to release its resources.
                logger.exception("fatal teardown failed for request %s", request_id)

    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        if self._fatal_error is not None:
            raise self._fatal_error
        return self.runtime_controller.step()

    def stats(self) -> dict[str, Any]:
        return self.runtime_controller.stats()

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.runtime_controller.reset_stats(clear_prefix_cache=clear_prefix_cache)
