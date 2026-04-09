# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import os
import re
import torch
import torch.nn as nn
import copy
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
from vllm.engine.runtime_factory import LiteRuntimeFactory
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


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expands seq_lens and block_tables for PagedAttention kernels during prefill.
    Standardizes 'Chunked Prefill' logic across Llama and Qwen architectures.
    """
    if seq > 1 and is_prefill:
        if bs == 1:
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
                block_tables_ext_parts.append(
                    block_tables[bi : bi + 1].expand(seq, -1)
                )
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
    sched_seqs = getattr(vllm_config.scheduler_config, "max_num_seqs", execution_policy_max)
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
        self.runtime_config = (
            getattr(vllm_config, "runtime_config", None)
            or RuntimeConfig.from_vllm_config(vllm_config)
        )
        requested_policy_mode = self.runtime_config.policy_mode
        
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
        )

        planner = RuntimePlanner(self.runtime_config, self.model_capabilities)
        execution_plan = planner.build_execution_plan(
            self.execution_policy.max_active_requests
        )
        kv_plan = planner.build_kv_cache_plan(execution_plan)
        gpu_total_gb = get_total_gpu_memory_gb()
        is_high_end_gpu = execution_plan.is_high_end_gpu
        if is_high_end_gpu:
            print(f">>>> LiteEngine: High-end GPU detected ({gpu_total_gb:.1f}GB). Enabling aggressive optimization.")

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
            print(">>>> LiteEngine: KV Cache quantized to TurboQuant INT4 (uint8 packed) [NEW]")
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
            print(f">>>> LiteEngine: Using KV scales: K={self.inf_config.k_scale}, V={self.inf_config.v_scale}")
        elif kv_plan.kv_dtype == torch.float8_e4m3fn:
            self.inf_config.kv_type = "fp8"
            print(f">>>> LiteEngine: KV Cache quantized to FP8 (e4m3fn) [STABLE]")
            self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
        else:
            self.inf_config.kv_type = "fp16"
            print(">>>> LiteEngine: KV Cache using full precision (BF16/FP16) [ACCURATE]")
            if self.model_capabilities.preferred_kv_dtype == "bfloat16":
                print(">>>> LiteEngine: KV Cache dtype bfloat16 (Qwen3.5)")
                self.kv_dtype = kv_plan.kv_dtype
            else:
                print(">>>> LiteEngine: KV Cache dtype float16")
                self.kv_dtype = kv_plan.kv_dtype
            self.kv_head_dim = kv_plan.kv_head_dim
            
        kv_theory_bytes = kv_plan.theory_bytes
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
            # Shape: (num_total_blocks, block_size, heads, head_size)
            k = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, self.kv_head_dim), 
                          device=self.device, dtype=self.kv_dtype)
            v = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, self.kv_head_dim), 
                          device=self.device, dtype=self.kv_dtype)
            self.kv_caches.append((k, v))
        
        if kv_plan.needs_scale_cache:
            print(">>>> LiteEngine: Allocating KV Scale Caches for TurboQuant...")
            self.kv_scale_caches = []
            for _ in range(self.num_layers):
                # Per-token, per-head scale: (num_total_blocks, block_size, num_kv_heads, 1)
                ks = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, 1), 
                               device=self.device, dtype=torch.float32)
                vs = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, 1), 
                               device=self.device, dtype=torch.float32)
                self.kv_scale_caches.append((ks, vs))
        else:
            self.kv_scale_caches = [(None, None)] * self.num_layers

        print(">>>> LiteEngine: KV Cache allocated successfully.")

        mem_after_kv = int(torch.cuda.memory_allocated(self.device))
        kv_delta_bytes = mem_after_kv - mem_before_kv
        total_gb = mem_after_kv / (1024**3)
        weights_gb = mem_before_kv / (1024**3)
        kv_delta_gb = kv_delta_bytes / (1024**3)
        gpu_total_gb = get_total_gpu_memory_gb()
        print(
            ">>>> LiteEngine: GPU memory breakdown (torch.cuda.memory_allocated; "
            "host RSS not included — large GGUF load is often CPU anon-rss):"
        )
        print(f"     before_KV (weights + overhead): {weights_gb:.3f} GiB")
        print(f"     KV pool (delta alloc):          {kv_delta_gb:.3f} GiB  (theory {kv_theory_bytes / (1024**3):.3f} GiB)")
        print(f"     after_KV total:                 {total_gb:.3f} GiB  /  GPU cap ~{gpu_total_gb:.1f} GiB")
        if gpu_total_gb > 0 and total_gb > 0.85 * gpu_total_gb:
            print(
                "     [Warn] Total allocated is high vs GPU size; reduce FASTINFERENCE_KV_MAX_MODEL_LEN "
                "or FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS, or use FASTINFERENCE_KV_FP8=1, or --frugal scheduling."
            )

        # slot_mapping maps batch tokens to physical indices
        self.scheduler = RequestScheduler(self.max_active_requests)
        self.lora_registry = LoRARuntimeRegistry()
        self.policies = None
        self.sampling_driver = None
        self.output_pipeline = None
        self.request_builder = None
        self.observer = getattr(vllm_config, "runtime_observer", None) or NullRuntimeObserver()
        self._queue_timeout_s = float(
            os.environ.get("FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS", "30.0")
        )
        
        # Pre-allocate tensors for SYNC FAST PATH (BS=1 to max_active_requests)
        # These will be reused to avoid Python object creation in every decode step.
        self._fast_input_ids = torch.empty((self.max_active_requests, 1), dtype=torch.long, device=self.device)
        self._fast_positions = torch.empty((self.max_active_requests, 1), dtype=torch.long, device=self.device)
        self._fast_slot_mapping = torch.empty((self.max_active_requests,), dtype=torch.long, device=self.device)
        self._fast_seq_lens = torch.empty((self.max_active_requests,), dtype=torch.int32, device=self.device)
        self._fast_block_tables = torch.empty((self.max_active_requests, self.num_blocks_per_seq), dtype=torch.int32, device=self.device)
        
        # Static block tables (only depends on slot_idx)
        for s in range(self.max_active_requests):
            start_block = s * self.num_blocks_per_seq
            self._fast_block_tables[s] = torch.arange(start_block, start_block + self.num_blocks_per_seq, dtype=torch.int32, device=self.device)
        runtime_components = LiteRuntimeFactory.build(self)
        self.kv_block_manager = runtime_components["kv_block_manager"]
        self.input_batch_builder = runtime_components["input_batch_builder"]
        self.prefill_executor = runtime_components["prefill_executor"]
        self.decode_executor = runtime_components["decode_executor"]
        self.step_scheduler = runtime_components["step_scheduler"]
        self.execution_backend = runtime_components["execution_backend"]
        self.runtime_controller = runtime_components["runtime_controller"]

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
                lora_int_id=resolved_lora.lora_int_id if resolved_lora is not None else None,
                lora_path=resolved_lora.lora_path if resolved_lora is not None else None,
                multi_modal_data=multi_modal_data,
            )
            self.multimodal_processor.prepare_request(request_state)
        except ValueError as exc:
            reason = (
                str(exc)
            )
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
                request_id, exc if isinstance(exc, BackgroundLoopError) else BackgroundLoopError(str(exc))
            )
            self.scheduler.free_request(request_id)

    @torch.inference_mode()
    def step(self) -> List[RequestOutput]:
        return self.runtime_controller.step()

    def stats(self) -> Dict[str, Any]:
        return self.runtime_controller.stats()

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.runtime_controller.reset_stats(clear_prefix_cache=clear_prefix_cache)
