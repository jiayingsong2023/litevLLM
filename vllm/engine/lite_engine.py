# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import os
import re
import torch
import torch.nn as nn
import copy
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from vllm.adapters import get_model_adapter
from vllm.config import VllmConfig
from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.errors import BackgroundLoopError, RequestRejectedError
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb, select_loadtime_policy
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.lora_runtime import LoRARuntimeRegistry
from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.engine.request_builder import LiteRequestBuilder
from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext
from vllm.engine.runtime_observer import NullRuntimeObserver
from vllm.engine.runtime_config import RuntimeConfig
from vllm.engine.runtime_planner import RuntimePlanner
from vllm.engine.sampling_driver import SamplingDriver
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.outputs import RequestOutput
from vllm.policies import build_generation_policies
from vllm.sampling_params import SamplingParams
from vllm.engine.inference_config import LiteInferenceConfig

logger = init_logger(__name__)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is not None and dtype == f8:
        return 1
    if dtype == torch.uint8:
        return 1
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return 2


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _align_kv_ctx_len(ctx: int, block_size: int, floor: int = 256) -> int:
    ctx = max(floor, int(ctx))
    return max(block_size, (ctx // block_size) * block_size)


def expand_metadata_for_paged_attention(
    bs: int,
    seq: int,
    is_prefill: bool,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    q_device: torch.device,
    seq_lens_cpu: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _resolve_kv_max_model_len(
    model_config: Any,
    vllm_config: VllmConfig,
    block_size: int,
) -> int:
    """Cap KV / slot stride by model config, scheduler, and optional env."""
    mc = model_config.get_max_model_len()
    sched = getattr(vllm_config.scheduler_config, "max_model_len", mc)
    cap = min(int(mc), int(sched), 4096)
    env = os.environ.get("FASTINFERENCE_KV_MAX_MODEL_LEN", "").strip()
    if env:
        cap = min(cap, max(block_size, int(env)))
    return _align_kv_ctx_len(cap, block_size)


def _resolve_kv_max_active_requests(
    execution_policy_max: int,
    vllm_config: VllmConfig,
) -> int:
    """Match paged KV pool to scheduler concurrency (and optional env)."""
    sched_seqs = getattr(
        vllm_config.scheduler_config, "max_num_seqs", execution_policy_max
    )
    out = min(int(execution_policy_max), int(sched_seqs))
    env = os.environ.get("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "").strip()
    if env:
        out = min(out, max(1, int(env)))
    return max(1, out)


class LiteEngine:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = torch.device("cuda:0")
        self.runtime_config = getattr(
            vllm_config, "runtime_config", None
        ) or RuntimeConfig.from_vllm_config(vllm_config)
        requested_policy_mode = self.runtime_config.policy_mode
        self.adapter = get_model_adapter(None, self.model_config)
        self.runtime_policy = self.adapter.runtime_policy(
            self.model_config,
            self.runtime_config,
        )
        self._install_tuning_configs_for_model(self.runtime_policy)

        # 1. Load Model
        print(f">>> LiteEngine: Loading {self.model_config.model}...")
        self.model = get_model(vllm_config=self.vllm_config)
        print(f">>> LiteEngine: Model Type: {type(self.model)}")
        self.tokenizer = None
        self.execution_policy = select_loadtime_policy(
            model_config=self.model_config,
            quant_config=getattr(vllm_config, "quant_config", None),
            policy_mode=requested_policy_mode,  # type: ignore[arg-type]
        )
        self.adapter = get_model_adapter(self.model, self.model_config)
        self.runtime_policy = self.adapter.runtime_policy(
            self.model_config,
            self.runtime_config,
        )
        self.model_capabilities = self.adapter.detect(self.model, self.model_config)
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
        self._apply_runtime_model_policy()
        if "FASTINFERENCE_AWQ_FUSED_SCOPE" in self._active_tuning_env:
            fused_stage = self._active_tuning_env.get(
                "FASTINFERENCE_AWQ_FUSED_SCOPE", "all"
            )
            print(
                ">>>> LiteEngine: fused rollout stage="
                f"{fused_stage} "
                "(set FASTINFERENCE_GEMMA4_FUSED_STAGE=off|attention_only|all to override)."
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
            kv_select_ratio=self.runtime_config.kv_select_ratio,
            kv_select_sig_dim=self.runtime_config.kv_select_sig_dim,
            kv_select_min_blocks=self.runtime_config.kv_select_min_blocks,
            kv_select_min_context=self.runtime_config.kv_select_min_context,
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
            print(f">>>> LiteEngine: KV Cache quantized to FP8 (e4m3fn) [STABLE]")
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

        # Allocate signature cache for KV Block Selective Attention.
        # sig_dim elements per (block, head), fp16.
        # Zero-initialized; populated lazily as blocks fill.
        sig_dim = int(getattr(self.inf_config, "kv_select_sig_dim", 32))
        self.sig_caches: list[torch.Tensor] = []
        self._sig_temp_buffers: list[torch.Tensor] = []
        _sig_enabled = self.inf_config.kv_select_ratio > 0.0
        for i in range(self.num_layers):
            layer_num_kv_heads, _layer_kv_head_dim = (
                self._layer_kv_cache_shape_for_layer(i)
            )
            if _sig_enabled:
                sig = torch.zeros(
                    (self.num_total_blocks, layer_num_kv_heads, sig_dim),
                    device=self.device,
                    dtype=torch.float16,
                )
                sig_temp = torch.zeros(
                    (
                        self.num_total_blocks,
                        self.block_size,
                        layer_num_kv_heads,
                        sig_dim,
                    ),
                    device=self.device,
                    dtype=torch.float16,
                )
            else:
                sig = torch.empty((0,), device=self.device, dtype=torch.float16)
                sig_temp = torch.empty((0,), device=self.device, dtype=torch.float16)
            self.sig_caches.append(sig)
            self._sig_temp_buffers.append(sig_temp)

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
                "     [Warn] Total allocated is high vs GPU size; reduce FASTINFERENCE_KV_MAX_MODEL_LEN "
                "or FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS, or use FASTINFERENCE_KV_TYPE=fp8, or --frugal scheduling."
            )

        # slot_mapping maps batch tokens to physical indices
        self.scheduler = RequestScheduler(self.max_active_requests)
        self.lora_registry = LoRARuntimeRegistry()
        self.policies = None
        self.sampling_driver = None
        self.output_pipeline = None
        self.request_builder = None
        self.observer = (
            getattr(vllm_config, "runtime_observer", None) or NullRuntimeObserver()
        )
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
            max_active_requests=self.max_active_requests,
            scheduler_policy=scheduler_policy,
            backend_policy=backend_policy,
            scheduler=self.scheduler,
            observer=self.observer,
            lora_registry=self.lora_registry,
            sampling_driver=self.sampling_driver,
            output_pipeline=self.output_pipeline,
            queue_timeout_s=self._queue_timeout_s,
            sig_caches=getattr(self, "sig_caches", None),
            sig_temp_buffers=getattr(self, "_sig_temp_buffers", None),
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
        param_dtype_bytes: Dict[str, int] = {}
        buffer_dtype_bytes: Dict[str, int] = {}
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

    def _resolve_layer_kv_specs(self) -> Optional[list[tuple[int, int]]]:
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
                nkv = int(getattr(attn, "num_kv_heads"))
                hdim = int(getattr(attn, "head_dim"))
                if nkv <= 0 or hdim <= 0:
                    return None
                specs.append((nkv, hdim))
            if len(specs) != int(self.num_layers):
                return None
            return specs
        except Exception:
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
                * _dtype_nbytes(self.kv_dtype)
            )
            if not needs_scale_cache:
                return data
            return data + int(
                self.num_layers
                * 2
                * self.num_total_blocks
                * self.block_size
                * self.num_kv_heads
                * _dtype_nbytes(torch.float32)
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
                * _dtype_nbytes(self.kv_dtype)
            )
            scale += (
                2
                * self.num_total_blocks
                * self.block_size
                * nkv
                * _dtype_nbytes(torch.float32)
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
                * _dtype_nbytes(torch.float32)
            )
        total = 0
        for i in range(self.num_layers):
            nkv, _cache_hdim = self._layer_kv_cache_shape_for_layer(i)
            total += (
                2
                * self.num_total_blocks
                * self.block_size
                * nkv
                * _dtype_nbytes(torch.float32)
            )
        return int(total)

    @property
    def active_request_count(self) -> int:
        """Number of in-flight requests (for debugging / test harness guards)."""
        return self.scheduler.active_request_count

    @staticmethod
    def _stack_per_layer_carries(
        req_dicts: List[Dict[str, Any]], num_layers: int, key: str
    ) -> List[Optional[torch.Tensor]]:
        """Batch (B, ...) tensors per layer for Qwen3.5 linear-attn streaming state."""
        stacked: List[Optional[torch.Tensor]] = []
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
        stacked: List[Optional[torch.Tensor]],
        req_dicts: List[Dict[str, Any]],
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

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_id: Optional[str] = None,
        lora_request: Optional[Any] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
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
    def step(self) -> List[RequestOutput]:
        return self.runtime_controller.step()

    def stats(self) -> Dict[str, Any]:
        return self.runtime_controller.stats()

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.runtime_controller.reset_stats(clear_prefix_cache=clear_prefix_cache)
