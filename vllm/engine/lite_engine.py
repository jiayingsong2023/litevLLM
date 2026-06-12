# SPDX-License-Identifier: Apache-2.0
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from vllm.adapters import get_model_adapter
from vllm.config import VllmConfig
from vllm.engine.errors import BackgroundLoopError, RequestRejectedError
from vllm.engine.inference_config import LiteInferenceConfig
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb, select_loadtime_policy
from vllm.engine.lora_runtime import LoRARuntimeRegistry
from vllm.engine.model_surface import resolve_model_surface
from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.request_builder import LiteRequestBuilder
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.runtime_config import RuntimeConfig
from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext
from vllm.engine.runtime_observer import NullRuntimeObserver
from vllm.engine.runtime_planner import RuntimePlanner
from vllm.engine.sampling_driver import SamplingDriver
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.policies import build_generation_policies
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import dtype_nbytes

logger = init_logger(__name__)


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = torch.device("cuda:0")
        self.runtime_config = getattr(
            vllm_config, "runtime_config", None
        ) or RuntimeConfig.from_vllm_config(vllm_config)
        self.vllm_config.runtime_config = self.runtime_config
        requested_policy_mode = self.runtime_config.policy_mode
        self.adapter = get_model_adapter(None, self.model_config)
        self.runtime_policy = self.adapter.runtime_policy(
            self.model_config,
            self.runtime_config,
        )
        self._apply_runtime_model_policy()
        self._install_runtime_policy_on_runtime_config(self.runtime_policy)
        self.vllm_config.runtime_config = self.runtime_config
        self._install_tuning_configs_for_model(self.runtime_policy)

        # 1. Load Model
        print(f">>> LiteEngine: Loading {self.model_config.model}...")
        self.model = get_model(vllm_config=self.vllm_config)
        print(f">>> LiteEngine: Model Type: {type(self.model)}")
        self.tokenizer = None
        self._deepseek_v4_flash_direct = isinstance(
            self.model,
            DeepSeekV4FlashForCausalLM,
        )
        if self._deepseek_v4_flash_direct:
            self._install_deepseek_v4_flash_direct_runtime()
            return
        self.execution_policy = select_loadtime_policy(
            model_config=self.model_config,
            quant_config=getattr(vllm_config, "quant_config", None),
            policy_mode=requested_policy_mode,  # type: ignore[arg-type]
        )
        self.adapter = get_model_adapter(self.model, self.model_config)
        self.model_capabilities = self.adapter.detect(self.model, self.model_config)
        self.model_surface = resolve_model_surface(
            model_name=str(getattr(self.model_config, "model", "")),
            capabilities=self.model_capabilities,
        )
        self.vllm_config.model_capabilities = self.model_capabilities
        self.num_attention_heads = self.model_capabilities.num_attention_heads
        self.num_kv_heads = self.model_capabilities.num_kv_heads
        self.head_size = self.model_capabilities.head_dim
        self.num_layers = self.model_capabilities.num_layers
        print(
            f">>> LiteEngine: Verified Dimensions: {self.num_attention_heads} Q-heads, "
            f"{self.num_kv_heads} KV-heads, {self.head_size} head_dim"
        )
        self._layer_kv_specs = self._resolve_layer_kv_specs()
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

        # 3. Pre-allocate Block-based KV Cache (paged: block table + fixed pool; block_size tokens/block)
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
            gemma4_c1_preset=self.runtime_config.gemma4_c1_preset,
            tuning_env=self._active_tuning_env,
            model_policy=dict(self.runtime_policy.model_policy),
            kernel_policy=dict(self.runtime_policy.kernel_policy),
            kv_select_ratio=self.runtime_config.kv_select_ratio,
            kv_select_min_blocks=self.runtime_config.kv_select_min_blocks,
        )

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
                f">>>> LiteEngine: High-end GPU detected ({gpu_total_gb:.1f}GB). Enabling aggressive optimization."
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

        print(
            ">>>> LiteEngine: Step scheduler "
            f"(token_budget={self._step_token_budget}, decode_priority={self._decode_priority_enabled}, "
            f"prefill_reserved_tokens={self._prefill_reserved_tokens}, "
            f"prefill_reserve_backlog={self._prefill_reserve_backlog}, "
            f"prefill_catchup_ratio={self._prefill_catchup_ratio:.2f}, "
            f"prefill_microbatch={self._prefill_microbatch_size})"
        )

        # Resolve KV Metadata from config
        if kv_plan.kv_dtype == torch.uint8:
            self.inf_config.kv_type = "turbo_int4"
            print(
                ">>>> LiteEngine: KV Cache quantized to TurboQuant INT4 (uint8 packed) [NEW]"
            )
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
            print(
                f">>>> LiteEngine: Using KV scales: K={self.inf_config.k_scale}, V={self.inf_config.v_scale}"
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
                self.kv_dtype = kv_plan.kv_dtype
            else:
                print(">>>> LiteEngine: KV Cache dtype float16")
                self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim

        kv_theory_bytes = kv_plan.theory_bytes
        if self._layer_kv_specs is not None:
            kv_theory_bytes = self._compute_kv_theory_bytes(
                needs_scale_cache=kv_plan.needs_scale_cache
            )
        print(
            f">>>> LiteEngine: Allocating KV Cache on {self.device} "
            f"({self.max_active_requests} seq slots, {self.max_model_len} tokens/seq cap, "
            f"{self.num_layers} layers, block={self.block_size} tok, dtype={self.kv_dtype}, "
            f"~{kv_theory_bytes / (1024**3):.3f} GiB theoretical)"
        )

        mem_before_kv = int(torch.cuda.memory_allocated(self.device))

        self.kv_caches = []
        for i in range(self.num_layers):
            print(f"    Allocating layer {i}...")
            layer_num_kv_heads, layer_kv_head_dim = (
                self._layer_kv_cache_shape_for_layer(i)
            )
            # Shape: (num_total_blocks, block_size, heads, head_size)
            k = torch.zeros(
                (
                    self.num_total_blocks,
                    self.block_size,
                    layer_num_kv_heads,
                    layer_kv_head_dim,
                ),
                device=self.device,
                dtype=self.kv_dtype,
            )
            v = torch.zeros(
                (
                    self.num_total_blocks,
                    self.block_size,
                    layer_num_kv_heads,
                    layer_kv_head_dim,
                ),
                device=self.device,
                dtype=self.kv_dtype,
            )
            self.kv_caches.append((k, v))

        if kv_plan.needs_scale_cache:
            print(">>>> LiteEngine: Allocating KV Scale Caches for TurboQuant...")
            self.kv_scale_caches = []
            for i in range(self.num_layers):
                layer_num_kv_heads, _layer_kv_head_dim = (
                    self._layer_kv_cache_shape_for_layer(i)
                )
                # Per-token, per-head scale: (num_total_blocks, block_size, num_kv_heads, 1)
                ks = torch.zeros(
                    (self.num_total_blocks, self.block_size, layer_num_kv_heads, 1),
                    device=self.device,
                    dtype=torch.float32,
                )
                vs = torch.zeros(
                    (self.num_total_blocks, self.block_size, layer_num_kv_heads, 1),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.kv_scale_caches.append((ks, vs))
        else:
            self.kv_scale_caches = [(None, None)] * self.num_layers

        print(">>>> LiteEngine: KV Cache allocated successfully.")

        mem_after_kv = int(torch.cuda.memory_allocated(self.device))
        kv_delta_bytes = mem_after_kv - mem_before_kv
        total_gb = _bytes_to_gib(mem_after_kv)
        weights_gb = _bytes_to_gib(mem_before_kv)
        kv_delta_gb = _bytes_to_gib(kv_delta_bytes)
        gpu_total_gb = get_total_gpu_memory_gb()
        audit = self._collect_cuda_tensor_memory_audit()
        params_total_bytes = int(audit["params_total_bytes"])
        buffers_total_bytes = int(audit["buffers_total_bytes"])
        awq_cache_bytes = int(audit["awq_cache_bytes"])
        accounted_before_kv = params_total_bytes + buffers_total_bytes + awq_cache_bytes
        other_cuda_overhead = max(0, mem_before_kv - accounted_before_kv)
        kv_data_theory = self._compute_kv_theory_bytes(needs_scale_cache=False)
        kv_scale_theory = (
            self._compute_kv_scale_theory_bytes() if kv_plan.needs_scale_cache else 0
        )
        print(
            ">>>> LiteEngine: GPU memory breakdown (torch.cuda.memory_allocated; "
            "host RSS not included — large GGUF load is often CPU anon-rss):"
        )
        print(f"     before_KV (weights + overhead): {weights_gb:.3f} GiB")
        print(
            f"     KV pool (delta alloc):          {kv_delta_gb:.3f} GiB  (theory {kv_theory_bytes / (1024**3):.3f} GiB)"
        )
        print(
            f"     after_KV total:                 {total_gb:.3f} GiB  /  GPU cap ~{gpu_total_gb:.1f} GiB"
        )
        print(">>>> LiteEngine: Startup memory audit (CUDA tensors):")
        print(
            "     model params:                  "
            f"{_bytes_to_gib(params_total_bytes):.3f} GiB  ({audit['params_count']} tensors)"
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
            f"{_bytes_to_gib(buffers_total_bytes):.3f} GiB  ({audit['buffers_count']} tensors)"
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

        # slot_mapping maps batch tokens to physical indices
        self.scheduler = RequestScheduler(self.max_active_requests)
        self.scheduler.runtime_config = self.runtime_config
        self.lora_registry = LoRARuntimeRegistry()
        self.policies = None
        self.sampling_driver = None
        self.output_pipeline = None
        self.request_builder = None
        self.observer = (
            getattr(vllm_config, "runtime_observer", None) or NullRuntimeObserver()
        )
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
        self._fast_block_tables = torch.empty(
            (self.max_active_requests, self.num_blocks_per_seq),
            dtype=torch.int32,
            device=self.device,
        )

        # Static block tables (only depends on slot_idx)
        for s in range(self.max_active_requests):
            start_block = s * self.num_blocks_per_seq
            self._fast_block_tables[s] = torch.arange(
                start_block,
                start_block + self.num_blocks_per_seq,
                dtype=torch.int32,
                device=self.device,
            )
        scheduler_policy = self.runtime_config.scheduler_policy
        backend_policy = self.runtime_config.backend_policy
        runtime_context = RuntimeAssemblyContext(
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
            fast_block_tables=self._fast_block_tables,
            step_token_budget=self._step_token_budget,
            decode_priority_enabled=self._decode_priority_enabled,
            prefill_chunk_size=self._prefill_chunk_size,
            prefill_reserved_tokens=self._prefill_reserved_tokens,
            prefill_reserve_backlog=self._prefill_reserve_backlog,
            prefill_catchup_ratio=self._prefill_catchup_ratio,
            prefill_microbatch_size=self._prefill_microbatch_size,
            min_prefill_chunk_size=self.runtime_config.min_prefill_chunk_size,
            max_prefill_chunk_size=self.runtime_config.max_prefill_chunk_size,
            prefill_sla_ttft_ms=self.runtime_config.prefill_sla_ttft_ms,
            max_active_requests=self.max_active_requests,
            scheduler_policy=scheduler_policy,
            backend_policy=backend_policy,
            scheduler=self.scheduler,
            observer=self.observer,
            lora_registry=self.lora_registry,
            sampling_driver=self.sampling_driver,
            output_pipeline=self.output_pipeline,
            queue_timeout_s=self._queue_timeout_s,
        )
        runtime_components = LiteRuntimeFactory.build(runtime_context)
        self.kv_block_manager = runtime_components["kv_block_manager"]
        self.input_batch_builder = runtime_components["input_batch_builder"]
        self.multimodal_processor = runtime_components["multimodal_processor"]
        self.prefill_executor = runtime_components["prefill_executor"]
        self.decode_executor = runtime_components["decode_executor"]
        self.step_scheduler = runtime_components["step_scheduler"]
        self.execution_backend = runtime_components["execution_backend"]
        self.runtime_controller = runtime_components["runtime_controller"]

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

            set_awq_fused_tuning_config(tuning_env, locked=True)
        except Exception:
            logger.debug("Unable to install AWQ fused tuning config", exc_info=True)
        try:
            from vllm.model_executor.layers.quantization.tensor import (
                set_awq_tensor_tuning_config,
            )

            set_awq_tensor_tuning_config(tuning_env, locked=True)
        except Exception:
            logger.debug("Unable to install AWQ tensor tuning config", exc_info=True)
        try:
            self.adapter.install_tuning_config(tuning_env)
        except Exception:
            logger.debug("Unable to install model tuning config", exc_info=True)

    def _collect_cuda_tensor_memory_audit(self) -> dict[str, Any]:
        """
        Snapshot CUDA-resident model tensor footprint by dtype.
        This is startup-only diagnostics, not used in the hot path.
        """
        device = self.device
        topn = int(self.runtime_config.memory_audit_topn)
        param_total = 0
        buffer_total = 0
        param_count = 0
        buffer_count = 0
        param_dtype_bytes: dict[str, int] = {}
        buffer_dtype_bytes: dict[str, int] = {}
        param_rows: list[dict[str, Any]] = []
        buffer_rows: list[dict[str, Any]] = []

        for name, p in self.model.named_parameters():
            if not isinstance(p, torch.Tensor) or p.device != device:
                continue
            size = int(p.numel() * p.element_size())
            param_total += size
            param_count += 1
            key = _dtype_name(p.dtype)
            param_dtype_bytes[key] = param_dtype_bytes.get(key, 0) + size
            param_rows.append(
                {
                    "name": str(name),
                    "shape": tuple(int(x) for x in p.shape),
                    "dtype": key,
                    "bytes": int(size),
                }
            )

        for name, b in self.model.named_buffers():
            if not isinstance(b, torch.Tensor) or b.device != device:
                continue
            size = int(b.numel() * b.element_size())
            buffer_total += size
            buffer_count += 1
            key = _dtype_name(b.dtype)
            buffer_dtype_bytes[key] = buffer_dtype_bytes.get(key, 0) + size
            buffer_rows.append(
                {
                    "name": str(name),
                    "shape": tuple(int(x) for x in b.shape),
                    "dtype": key,
                    "bytes": int(size),
                }
            )

        awq_cache_bytes = 0
        try:
            from vllm.model_executor.layers.quantization.tensor import (
                get_awq_runtime_stats,
            )

            awq_stats = get_awq_runtime_stats()
            awq_cache_bytes = int(awq_stats.get("cache_bytes", 0) or 0)
        except Exception:
            logger.debug("AWQ runtime stats not available", exc_info=True)
            awq_cache_bytes = 0

        return {
            "params_total_bytes": int(param_total),
            "buffers_total_bytes": int(buffer_total),
            "params_count": int(param_count),
            "buffers_count": int(buffer_count),
            "params_dtype_bytes": param_dtype_bytes,
            "buffers_dtype_bytes": buffer_dtype_bytes,
            "awq_cache_bytes": int(awq_cache_bytes),
            "topn": int(topn),
            "params_top": sorted(param_rows, key=lambda x: -int(x["bytes"]))[:topn],
            "buffers_top": sorted(buffer_rows, key=lambda x: -int(x["bytes"]))[:topn],
        }

    def _resolve_layer_kv_specs(self) -> list[tuple[int, int]] | None:
        """
        Best-effort per-layer KV specs in unpacked domain: (num_kv_heads, head_dim).
        Falls back to model-capability-wide uniform dimensions when model internals are not inspectable.
        """
        try:
            layers = list(getattr(getattr(self.model, "model", None), "layers", []))
            if not layers:
                return None
            specs: list[tuple[int, int]] = []
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    return None
                nkv = int(attn.num_kv_heads)
                hdim = int(attn.head_dim)
                if nkv <= 0 or hdim <= 0:
                    return None
                specs.append((nkv, hdim))
            if len(specs) != int(self.num_layers):
                return None
            return specs
        except Exception:
            logger.debug(
                "_resolve_layer_kv_specs: best-effort fallback, "
                "per-layer KV specs not available",
            )
            return None

    def _layer_kv_cache_shape_for_layer(self, layer_idx: int) -> tuple[int, int]:
        if self._layer_kv_specs is None:
            return int(self.num_kv_heads), int(self.kv_head_dim)
        nkv, hdim = self._layer_kv_specs[layer_idx]
        if self.kv_dtype == torch.uint8:
            return int(nkv), int(hdim // 2)
        return int(nkv), int(hdim)

    def _compute_kv_theory_bytes(self, *, needs_scale_cache: bool) -> int:
        if self._layer_kv_specs is None:
            data = int(
                self.num_layers
                * 2
                * self.num_total_blocks
                * self.block_size
                * self.num_kv_heads
                * self.kv_head_dim
                * dtype_nbytes(self.kv_dtype)
            )
            if not needs_scale_cache:
                return data
            return data + int(
                self.num_layers
                * 2
                * self.num_total_blocks
                * self.block_size
                * self.num_kv_heads
                * dtype_nbytes(torch.float32)
            )
        data = 0
        scale = 0
        for i in range(self.num_layers):
            nkv, cache_hdim = self._layer_kv_cache_shape_for_layer(i)
            data += (
                2
                * self.num_total_blocks
                * self.block_size
                * nkv
                * cache_hdim
                * dtype_nbytes(self.kv_dtype)
            )
            scale += (
                2
                * self.num_total_blocks
                * self.block_size
                * nkv
                * dtype_nbytes(torch.float32)
            )
        return int(data + (scale if needs_scale_cache else 0))

    def _compute_kv_scale_theory_bytes(self) -> int:
        if self._layer_kv_specs is None:
            return int(
                self.num_layers
                * 2
                * self.num_total_blocks
                * self.block_size
                * self.num_kv_heads
                * dtype_nbytes(torch.float32)
            )
        total = 0
        for i in range(self.num_layers):
            nkv, _cache_hdim = self._layer_kv_cache_shape_for_layer(i)
            total += (
                2
                * self.num_total_blocks
                * self.block_size
                * nkv
                * dtype_nbytes(torch.float32)
            )
        return int(total)

    @property
    def active_request_count(self) -> int:
        """Number of in-flight requests (for debugging / test harness guards)."""
        return self.scheduler.active_request_count

    @staticmethod
    def _stack_per_layer_carries(
        req_dicts: list[dict[str, Any]], num_layers: int, key: str
    ) -> list[torch.Tensor | None]:
        """Batch (B, ...) tensors per layer for Qwen3.5 linear-attn streaming state."""
        stacked: list[torch.Tensor | None] = []
        for li in range(num_layers):
            parts = [r[key][li] for r in req_dicts]
            if all(p is None for p in parts):
                stacked.append(None)
            else:
                if any(p is None for p in parts):
                    raise RuntimeError(
                        f"Mixed None/non-None in batched decode for {key}[layer={li}]"
                    )
                # Each request stores (1, ...) slices; concatenate batch dim, do not stack
                # (stack would produce (B, 1, ...) and break Qwen3.5 conv carry cat).
                stacked.append(torch.cat(parts, dim=0))
        return stacked

    @staticmethod
    def _split_per_layer_carries(
        stacked: list[torch.Tensor | None],
        req_dicts: list[dict[str, Any]],
        key: str,
    ) -> None:
        for li, t in enumerate(stacked):
            for i, r in enumerate(req_dicts):
                if t is None:
                    r[key][li] = None
                else:
                    r[key][li] = t[i : i + 1].contiguous()

    def register_lora_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str | None = None,
        lora_int_id: int | None = None,
    ) -> dict[str, Any]:
        request = self.lora_registry.register_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
            lora_int_id=lora_int_id,
        )
        return {
            "lora_name": request.lora_name,
            "lora_int_id": request.lora_int_id,
            "lora_path": request.lora_path,
        }

    def unregister_lora_adapter(self, lora_name: str) -> bool:
        return self.lora_registry.unregister_adapter(lora_name)

    def _install_deepseek_v4_flash_direct_runtime(self) -> None:
        self.execution_policy = SimpleNamespace(max_tokens_cap=8192)
        self.model_capabilities = None
        self.model_surface = None
        self.num_attention_heads = 0
        self.num_kv_heads = 0
        self.head_size = 0
        self.num_layers = 0
        self.max_active_requests = 1
        self.max_model_len = int(getattr(self.model_config, "max_model_len", 4096))
        self.block_size = 0
        self.num_blocks_per_seq = 0
        self.num_total_blocks = 0
        self.kv_caches = []
        self.kv_scale_caches = []
        self.scheduler = None
        self.lora_registry = LoRARuntimeRegistry()
        self.policies = None
        self.sampling_driver = None
        self.output_pipeline = None
        self.request_builder = None
        self.observer = (
            getattr(self.vllm_config, "runtime_observer", None) or NullRuntimeObserver()
        )
        self._queue_timeout_s = float(
            getattr(self.runtime_config, "queue_timeout_s", 30.0)
        )
        self.runtime_controller = None

    def generate_deepseek_v4_flash_greedy(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> RequestOutput:
        if not getattr(self, "_deepseek_v4_flash_direct", False):
            raise RuntimeError("DeepSeek V4 Flash direct runtime is not enabled")
        if self.tokenizer is None:
            raise RuntimeError("DeepSeek V4 Flash direct runtime requires a tokenizer")
        self._validate_deepseek_v4_flash_sampling(sampling_params)
        max_tokens = int(sampling_params.max_tokens or 1)
        prompt_token_ids = [int(token) for token in self.tokenizer.encode(prompt)]
        if not prompt_token_ids:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            prompt_token_ids = [0 if eos is None else int(eos)]
        input_ids = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        output_ids = self.model.generate_greedy_kernel(
            input_ids,
            max_tokens=max_tokens,
        )
        generated_token_ids = [
            int(token)
            for token in output_ids[len(prompt_token_ids) :].detach().cpu().tolist()
        ]
        try:
            text = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=sampling_params.skip_special_tokens,
            )
        except TypeError:
            text = self.tokenizer.decode(generated_token_ids)
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=generated_token_ids,
                    cumulative_logprob=0.0,
                )
            ],
            finished=True,
        )

    @staticmethod
    def _validate_deepseek_v4_flash_sampling(
        sampling_params: SamplingParams,
    ) -> None:
        if int(getattr(sampling_params, "n", 1) or 1) != 1:
            raise ValueError("DeepSeek V4 Flash direct runtime supports n=1 only")
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens is None or int(max_tokens) <= 0:
            raise ValueError("DeepSeek V4 Flash direct runtime requires max_tokens > 0")
        if float(getattr(sampling_params, "temperature", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        if float(getattr(sampling_params, "top_p", 1.0) or 1.0) != 1.0:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        top_k = int(getattr(sampling_params, "top_k", -1) or -1)
        if top_k not in (-1, 0):
            raise ValueError(
                "DeepSeek V4 Flash direct runtime supports greedy sampling only"
            )
        if getattr(sampling_params, "structured_outputs", None) is not None:
            raise ValueError(
                "DeepSeek V4 Flash direct runtime does not support structured outputs"
            )

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_id: str | None = None,
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ):
        if self.policies is None:
            self.policies = build_generation_policies(
                str(self.model_config.model), self.tokenizer, self.adapter
            )
            self.sampling_driver = SamplingDriver(
                self.tokenizer,
                getattr(self.model_config, "hf_config", None),
                self.policies,
            )
            self.output_pipeline = OutputPipeline(
                self.tokenizer, self.policies, self.sampling_driver
            )
            self.request_builder = LiteRequestBuilder(
                tokenizer=self.tokenizer,
                policies=self.policies,
                device=self.device,
                num_layers=self.num_layers,
                max_model_len=self.max_model_len,
                max_tokens_cap=self.execution_policy.max_tokens_cap,
                default_min_new_tokens=self.runtime_config.default_min_new_tokens,
            )
            self.execution_backend.sampling_driver = self.sampling_driver
            self.execution_backend.output_coordinator = self.output_pipeline
        if not self.scheduler.has_queue_capacity():
            reason = (
                "request queue full "
                f"(running={self.scheduler.running_request_count}, queued={self.scheduler.queued_request_count})"
            )
            self.observer.on_request_rejected(request_id, reason)
            raise RequestRejectedError(reason)

        try:
            resolved_lora = self.lora_registry.resolve_adapter(
                lora_id=lora_id,
                lora_request=lora_request,
            )
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
            self.multimodal_processor.prepare_request(request_state)
        except ValueError as exc:
            reason = str(exc)
            self.observer.on_request_rejected(request_id, reason)
            raise RequestRejectedError(reason)
        self.execution_backend.maybe_apply_prefix_cache(request_state)
        self.scheduler.enqueue_request(request_id, request_state)
        self.lora_registry.on_request_added(request_state.get("lora_id"))
        self.observer.on_request_added(request_id, request_state)

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        async for output in self.scheduler.get_request_stream(request_id):
            yield output

    def abort_request(self, request_id: str) -> None:
        try:
            req = self.scheduler.get_request(request_id)
        except KeyError:
            return
        output = self.output_pipeline.build_abort_output(request_id, req)
        self.scheduler.publish_output(request_id, output)
        self.scheduler.abort_request(request_id)
        self.observer.on_request_aborted(request_id)

    def handle_background_error(self, exc: BaseException) -> None:
        request_ids = self.scheduler.request_ids()
        self.observer.on_background_error(exc, request_ids)
        for request_id in request_ids:
            self.scheduler.publish_exception(
                request_id,
                exc
                if isinstance(exc, BackgroundLoopError)
                else BackgroundLoopError(str(exc)),
            )
            self.scheduler.free_request(request_id)

    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        return self.runtime_controller.step()

    def stats(self) -> dict[str, Any]:
        if getattr(self, "_deepseek_v4_flash_direct", False):
            return {"backend": "deepseek_v4_flash_direct"}
        return self.runtime_controller.stats()

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        if getattr(self, "_deepseek_v4_flash_direct", False):
            return
        self.runtime_controller.reset_stats(clear_prefix_cache=clear_prefix_cache)
