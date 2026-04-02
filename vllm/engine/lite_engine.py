# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import os
import re
from collections import Counter
import torch
import torch.nn as nn
import copy
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from vllm.config import VllmConfig
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb, select_loadtime_policy
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.engine.inference_config import LiteInferenceConfig
from vllm.engine.output_processor import get_output_processor, _decode_generated_text

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


def _hf_config_eos_token_ids(hf_config: Optional[Any]) -> List[int]:
    """`config.json` / `text_config` may list eos_token_id(s) that differ from tokenizer.eos_token_id."""
    if hf_config is None:
        return []
    eos = getattr(hf_config, "eos_token_id", None)
    if eos is None:
        return []
    if isinstance(eos, (list, tuple)):
        return [int(x) for x in eos]
    return [int(eos)]


def _eos_stop_token_ids_for_sampling(
    tokenizer: Any,
    sp: SamplingParams,
    hf_config: Optional[Any] = None,
) -> List[int]:
    """
    Merge tokenizer EOS, HF config EOS (e.g. Qwen3.5 GGUF: text_config vs tokenizer), and
    user stop_token_ids so we stop / mask consistently with how the checkpoint was trained.
    """
    out: List[int] = []
    seen: set[int] = set()

    def _add(tid: int) -> None:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            for x in eos:
                _add(int(x))
        else:
            _add(int(eos))
    for tid in _hf_config_eos_token_ids(hf_config):
        _add(tid)
    for tid in getattr(sp, "stop_token_ids", None) or ():
        _add(int(tid))
    return out


def _apply_lite_default_min_tokens(sp: SamplingParams, max_new_tokens: int) -> None:
    """
    Optional bump when callers leave min_tokens at 0 (SamplingParams default).
    Default is off (same as explicit min_tokens=0: immediate EOS allowed).
    Set FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS=1 (or higher) to reduce
    first-token EOS on chat-style models (e.g. Qwen3.5 + GGUF).
    """
    if int(getattr(sp, "min_tokens", 0) or 0) != 0:
        return
    raw = os.environ.get("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "0")
    if isinstance(raw, str):
        raw = raw.strip()
    if raw == "" or str(raw).lower() in ("0", "false", "off", "no"):
        return
    try:
        sp.min_tokens = max(0, int(raw))
    except ValueError:
        sp.min_tokens = 1
    mt = int(getattr(sp, "min_tokens", 0) or 0)
    if mt > max_new_tokens:
        sp.min_tokens = max(0, max_new_tokens)


def _sample_token_from_logits(
    logits_1d: torch.Tensor,
    sp: SamplingParams,
    generated_ids: List[int],
    generator: Optional[torch.Generator],
    eos_token_ids_to_mask: Optional[List[int]] = None,
    anti_template_token_ids: Optional[List[int]] = None,
) -> int:
    """
    Greedy argmax when temperature <= 0; otherwise temperature scaling, optional
    repetition_penalty, top-k / top-p filtering, then multinomial sampling.
    """
    logits = logits_1d.float().clone()
    rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
    if rp > 1.0 and generated_ids:
        for tid in set(int(t) for t in generated_ids):
            if 0 <= tid < logits.numel():
                if logits[tid] > 0:
                    logits[tid] /= rp
                else:
                    logits[tid] *= rp

    fp = float(getattr(sp, "frequency_penalty", 0.0) or 0.0)
    if abs(fp) > 1e-12 and generated_ids:
        for tid, cnt in Counter(int(t) for t in generated_ids).items():
            if 0 <= tid < logits.numel():
                logits[tid] -= fp * float(cnt)

    pp = float(getattr(sp, "presence_penalty", 0.0) or 0.0)
    if abs(pp) > 1e-12 and generated_ids:
        for tid in set(int(t) for t in generated_ids):
            if 0 <= tid < logits.numel():
                logits[tid] -= pp

    if not getattr(sp, "ignore_eos", False) and eos_token_ids_to_mask:
        mt = int(getattr(sp, "min_tokens", 0) or 0)
        if len(generated_ids) < mt:
            for tid in eos_token_ids_to_mask:
                if 0 <= tid < logits.numel():
                    logits[tid] = float("-inf")
    elif getattr(sp, "ignore_eos", False) and eos_token_ids_to_mask:
        for tid in eos_token_ids_to_mask:
            if 0 <= tid < logits.numel():
                logits[tid] = float("-inf")

    if anti_template_token_ids and len(generated_ids) < 12:
        for tid in anti_template_token_ids:
            if 0 <= tid < logits.numel():
                logits[tid] -= 60.0

    temp = float(getattr(sp, "temperature", 0.0) or 0.0)
    if temp <= 1e-6:
        return int(torch.argmax(logits).item())

    logits = logits / max(temp, 1e-6)

    top_k = int(getattr(sp, "top_k", -1) or -1)
    if 0 < top_k < logits.numel():
        k = min(top_k, logits.numel())
        _, top_idx = torch.topk(logits, k)
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep[top_idx] = True
        logits = logits.masked_fill(~keep, float("-inf"))

    top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
    if top_p < 1.0 - 1e-6:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumprobs > top_p
        sorted_remove[1:] = sorted_remove[:-1].clone()
        sorted_remove[0] = False
        if sorted_remove.all():
            sorted_remove[:] = True
            sorted_remove[0] = False
        removed_idx = sorted_indices[sorted_remove]
        logits = logits.clone()
        logits[removed_idx] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    psum = probs.sum()
    if not torch.isfinite(psum) or psum <= 0:
        return int(torch.argmax(logits_1d.float()).item())
    if generator is not None:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return int(torch.multinomial(probs, 1).item())


class LiteEngine:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = torch.device("cuda:0")
        requested_policy_mode = str(
            getattr(vllm_config, "runtime_policy_mode", "auto")
        ).lower()
        
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
        
        # 2. Extract REAL dimensions from loaded model
        try:
            # Prefer actual model attributes (set by loader)
            inner_model = getattr(self.model, "model", self.model)
            first_layer = None
            if hasattr(inner_model, "layers") and len(inner_model.layers) > 0:
                first_layer = inner_model.layers[0]
            
            if first_layer:
                # Standard attributes we set in llama.py / qwen3_5.py
                self.num_attention_heads = getattr(first_layer, "num_heads", 0)
                self.num_kv_heads = getattr(first_layer, "num_kv_heads", 0)
                self.head_size = getattr(first_layer, "head_dim", 0)
                
                # If not found, try nested self_attn
                if self.num_attention_heads == 0 and hasattr(first_layer, "self_attn"):
                    self.num_attention_heads = getattr(first_layer.self_attn, "num_heads", 0)
                    self.num_kv_heads = getattr(first_layer.self_attn, "num_kv_heads", 0)
                    self.head_size = getattr(first_layer.self_attn, "head_dim", 0)

            from vllm.model_executor.models.qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
                Qwen3_5MoeForConditionalGeneration,
            )
            is_qwen35 = isinstance(
                self.model,
                (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration),
            )

            # Fallback to config if model inspection failed, or force config for Qwen3.5 hybrid layers.
            if self.num_attention_heads == 0 or self.head_size == 0 or is_qwen35:
                self.num_kv_heads = self.model_config.get_num_kv_heads(None)
                self.head_size = self.model_config.get_head_size()
                hc = self.model_config.hf_config
                self.num_attention_heads = getattr(
                    hc, "num_attention_heads", self.num_kv_heads
                )
                hd = getattr(hc, "head_dim", None) or (
                    hc.get("head_dim") if isinstance(hc, dict) else None
                )
                if hd:
                    self.head_size = int(hd)
            
            # FORCE ALIGNMENT TO 8 FOR HARDWARE STABILITY
            if self.head_size % 8 != 0:
                old_hs = self.head_size
                self.head_size = (self.head_size + 7) // 8 * 8
                print(f">>> LiteEngine: Hard-aligning head_size {old_hs} -> {self.head_size} for stability")
            
            print(f">>> LiteEngine: Verified Dimensions: {self.num_attention_heads} Q-heads, {self.num_kv_heads} KV-heads, {self.head_size} head_dim")
        except Exception as e:
            print(f">>> LiteEngine: Dimension detection failed ({e}), using defaults")
            self.num_kv_heads = self.model_config.get_num_kv_heads(None)
            self.head_size = self.model_config.get_head_size()
            self.num_attention_heads = self.num_kv_heads

        self.num_layers = self.model_config.get_num_layers(None)
        
        # 3. Pre-allocate Block-based KV Cache (paged: block table + fixed pool; block_size tokens/block)
        self.inf_config = LiteInferenceConfig.from_env()

        gpu_total_gb = get_total_gpu_memory_gb()
        is_high_end_gpu = gpu_total_gb > 24.0
        if is_high_end_gpu:
            print(f">>>> LiteEngine: High-end GPU detected ({gpu_total_gb:.1f}GB). Enabling aggressive optimization.")

        # Use unified configuration
        pass

        self.block_size = self.inf_config.block_size
        self.max_model_len = _resolve_kv_max_model_len(self.model_config, self.vllm_config, self.block_size)
        if self.inf_config.max_model_len:
            self.max_model_len = _align_kv_ctx_len(min(self.max_model_len, self.inf_config.max_model_len), self.block_size)

        self.max_active_requests = _resolve_kv_max_active_requests(
            self.execution_policy.max_active_requests,
            self.vllm_config,
        )
        # Bumping default concurrency for 32GB+ cards
        if is_high_end_gpu and self.max_active_requests < 16 and os.environ.get("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS") is None:
            print(">>>> LiteEngine: Bumping max_active_requests 4 -> 16 for high-end GPU.")
            self.max_active_requests = 16

        if self.inf_config.max_active_requests != 4:
             self.max_active_requests = self.inf_config.max_active_requests

        self.num_blocks_per_seq = self.max_model_len // self.block_size
        self.num_total_blocks = self.max_active_requests * self.num_blocks_per_seq

        sched_token_budget = getattr(
            self.vllm_config.scheduler_config, "max_num_batched_tokens", None
        )
        # High-end GPU can handle more tokens per step
        default_budget = 8192 if is_high_end_gpu else 4096
        self._step_token_budget = max(1, int(sched_token_budget)) if sched_token_budget is not None else default_budget

        # Prefill chunk size optimization
        env_chunk = os.environ.get("FASTINFERENCE_LITE_PREFILL_CHUNK", "").strip()
        if env_chunk:
            env_chunk_int = int(env_chunk)
            self._prefill_chunk_size = env_chunk_int if env_chunk_int > 0 else self.max_model_len
        else:
            from vllm.model_executor.models.qwen3_5 import (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
            if isinstance(self.model, (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)):
                self._prefill_chunk_size = 2048 if is_high_end_gpu else 1024
            else:
                self._prefill_chunk_size = 1024 if is_high_end_gpu else 512

        self._decode_priority_enabled = _env_truthy("FASTINFERENCE_LITE_DECODE_PRIORITY") or os.environ.get("FASTINFERENCE_LITE_DECODE_PRIORITY", "").strip() == ""
        self._prefill_reserved_tokens = max(0, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS", "0")))
        self._prefill_reserve_backlog = max(1, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG", "2")))
        self._prefill_catchup_ratio = min(1.0, max(0.0, float(os.environ.get("FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO", "0.25"))))
        self._prefill_microbatch_size = min(4, max(1, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_MICROBATCH", "2"))))

        print(
            ">>>> LiteEngine: Step scheduler "
            f"(token_budget={self._step_token_budget}, decode_priority={self._decode_priority_enabled}, "
            f"prefill_reserved_tokens={self._prefill_reserved_tokens}, "
            f"prefill_reserve_backlog={self._prefill_reserve_backlog}, "
            f"prefill_catchup_ratio={self._prefill_catchup_ratio:.2f}, "
            f"prefill_microbatch={self._prefill_microbatch_size})"
        )
        
        # Ensure MoE models fallback to FP8 if TurboQuant is selected, due to extreme activation outliers
        from vllm.model_executor.models.qwen3_5 import (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
        if self.inf_config.kv_type == "turbo_int4" and hasattr(self, 'model') and isinstance(self.model, Qwen3_5MoeForConditionalGeneration):
            print(">>>> [Warning] TurboQuant INT4 KV cache does not support the massive activation outliers found in MoE networks. Automatically falling back to FP8.")
            self.inf_config.kv_type = "fp8"

        # Resolve KV Metadata from config
        if self.inf_config.kv_type == "turbo_int4":
            print(">>>> LiteEngine: KV Cache quantized to TurboQuant INT4 (uint8 packed) [NEW]")
            self.kv_dtype = torch.uint8
            self.kv_head_dim = self.head_size // 2
            print(f">>>> LiteEngine: Using KV scales: K={self.inf_config.k_scale}, V={self.inf_config.v_scale}")
        elif self.inf_config.kv_type == "fp8":
            print(f">>>> LiteEngine: KV Cache quantized to FP8 (e4m3fn) [STABLE]")
            self.kv_dtype = torch.float8_e4m3fn
            self.kv_head_dim = self.head_size
        else:
            print(">>>> LiteEngine: KV Cache using full precision (BF16/FP16) [ACCURATE]")
            from vllm.model_executor.models.qwen3_5 import (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
            if isinstance(self.model, (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)):
                print(">>>> LiteEngine: KV Cache dtype bfloat16 (Qwen3.5)")
                self.kv_dtype = torch.bfloat16
            else:
                print(">>>> LiteEngine: KV Cache dtype float16")
                self.kv_dtype = torch.float16
            self.kv_head_dim = self.head_size
            
        elem_nbytes = _dtype_nbytes(self.kv_dtype)
        kv_theory_bytes = (
            self.num_layers
            * 2
            * self.num_total_blocks
            * self.block_size
            * self.num_kv_heads
            * self.kv_head_dim
            * elem_nbytes
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
            # Shape: (num_total_blocks, block_size, heads, head_size)
            k = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, self.kv_head_dim), 
                          device=self.device, dtype=self.kv_dtype)
            v = torch.zeros((self.num_total_blocks, self.block_size, self.num_kv_heads, self.kv_head_dim), 
                          device=self.device, dtype=self.kv_dtype)
            self.kv_caches.append((k, v))
        
        if self.inf_config.kv_type == "turbo_int4":
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
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._running_ids: List[str] = []
        self._free_slots = list(range(self.max_active_requests))
        self._request_slots: Dict[str, int] = {} # Map req_id -> slot_idx
        self._request_streams: Dict[str, asyncio.Queue] = {}
        
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

    @property
    def active_request_count(self) -> int:
        """Number of in-flight requests (for debugging / test harness guards)."""
        return len(self._running_ids)

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

    def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams, lora_id: Optional[str] = None):
        if getattr(self, "output_processor", None) is None:
            self.output_processor = get_output_processor(str(self.model_config.model), self.tokenizer)
        if getattr(self.output_processor, "tokenizer", None) is None:
            self.output_processor.tokenizer = self.tokenizer
        if not self._free_slots:
            # Simple rejection if full. In real engine, we'd queue.
            print(f"!!! LiteEngine: Max capacity reached ({self.max_active_requests}), rejecting {request_id}")
            return

        effective_sampling_params = copy.deepcopy(sampling_params)
        max_tokens = effective_sampling_params.max_tokens or 16
        max_tokens = min(max_tokens, self.execution_policy.max_tokens_cap)
        _apply_lite_default_min_tokens(effective_sampling_params, max_tokens)

        guarded_prompt = self.output_processor.apply_prompt_guard(prompt)
        input_ids = self.tokenizer.encode(guarded_prompt)
        if len(input_ids) >= self.max_model_len:
            print(
                f"!!! LiteEngine: rejecting {request_id} because prompt tokens "
                f"({len(input_ids)}) exceed/equal max_model_len ({self.max_model_len}); "
                "leave at least one decode token slot."
            )
            return
        slot_idx = self._free_slots.pop(0)
        
        rng: Optional[torch.Generator] = None
        seed = getattr(effective_sampling_params, "seed", None)
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(seed))

        self._requests[request_id] = {
            "input_ids": input_ids,
            "generated_ids": [],
            "sampling_params": effective_sampling_params,
            "finished": False,
            "prompt": prompt,
            "guarded_prompt": guarded_prompt,
            "slot_idx": slot_idx,
            "seq_len": 0,  # Current length in KV cache
            "is_prefill": True,
            "lora_id": lora_id,
            "rng": rng,
            # Qwen3.5 linear-attn: recurrent delta-net state + causal conv tail (per layer).
            "linear_attn_carry": [None] * self.num_layers,
            "linear_conv_carry": [None] * self.num_layers,
            "low_info_hits": 0,
            "is_chinese_capital_question": self.output_processor.is_chinese_capital_question(prompt),
            "capital_question_bias_token_ids": self.output_processor.get_capital_question_bias_token_ids(prompt),
            "anti_template_token_ids": self.output_processor.get_anti_template_token_ids(),
        }
        self._request_slots[request_id] = slot_idx
        self._running_ids.append(request_id)
        self._request_streams[request_id] = asyncio.Queue()

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        queue = self._request_streams[request_id]
        while True:
            output = await queue.get()
            yield output
            if output.finished: break

    @torch.inference_mode()
    def _decode_step_sync(self, decodes: List[str]) -> List[RequestOutput]:
        bs = len(decodes)
        # Use sliced pre-allocated tensors
        input_ids = self._fast_input_ids[:bs]
        positions = self._fast_positions[:bs]
        slot_mapping = self._fast_slot_mapping[:bs]
        seq_lens = self._fast_seq_lens[:bs]
        
        # Populate pre-allocated tensors (minimal Python loops)
        req_dicts = []
        lora_mapping = []
        for i, rid in enumerate(decodes):
            req = self._requests[rid]
            req_dicts.append(req)
            input_ids[i, 0] = req["generated_ids"][-1]
            positions[i, 0] = req["seq_len"]
            slot_mapping[i] = req["slot_idx"] * self.max_model_len + req["seq_len"]
            seq_lens[i] = req["seq_len"] + 1
            lora_mapping.append(req.get("lora_id"))

        # Block tables are static, we just need to slice and gather
        # self._fast_block_tables has shape (max_active_requests, num_blocks_per_seq)
        # We need to select the rows corresponding to slot_indices.
        # This is fast using torch.index_select or advanced indexing.
        slots_t = torch.tensor([req["slot_idx"] for req in req_dicts], device=self.device)
        block_tables = self._fast_block_tables.index_select(0, slots_t)

        attn_carry_batch = self._stack_per_layer_carries(req_dicts, self.num_layers, "linear_attn_carry")
        conv_carry_batch = self._stack_per_layer_carries(req_dicts, self.num_layers, "linear_conv_carry")

        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "block_tables": block_tables,
            "is_prefill": False,
            "kv_start_indices": positions.squeeze(1).to(torch.int32),
            "linear_attn_carry": attn_carry_batch,
            "linear_conv_carry": conv_carry_batch,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
        }

        logits = self.model(input_ids, positions, self.kv_caches, attn_metadata, lora_mapping=lora_mapping)
        
        self._split_per_layer_carries(attn_metadata["linear_attn_carry"], req_dicts, "linear_attn_carry")
        self._split_per_layer_carries(attn_metadata["linear_conv_carry"], req_dicts, "linear_conv_carry")

        results = []
        for i, rid in enumerate(decodes):
            req = self._requests[rid]
            token_logits = self.output_processor.apply_context_bias(logits[i, -1, :], req['generated_ids'], req['sampling_params'], req.get('capital_question_bias_token_ids'), req.get('is_chinese_capital_question'))
            
            eos_mask = _eos_stop_token_ids_for_sampling(
                self.tokenizer, req["sampling_params"], getattr(self.model_config, "hf_config", None)
            )
            
            token = _sample_token_from_logits(
                token_logits, req["sampling_params"], req["generated_ids"], req.get("rng"),
                eos_token_ids_to_mask=eos_mask, anti_template_token_ids=req.get("anti_template_token_ids")
            )
            
            req["generated_ids"].append(token)
            req["seq_len"] += 1
            self._process_completion(rid, token, results)
            
        return results

    def _free_request(self, rid: str):
        if rid in self._requests:
            slot = self._requests[rid]["slot_idx"]
            self._free_slots.append(slot)
            # Optional: Clear KV cache for this slot? Not strictly necessary if we track seq_len correctly.
            del self._requests[rid]
        if rid in self._request_slots: del self._request_slots[rid]
        if rid in self._running_ids: self._running_ids.remove(rid)

    @torch.inference_mode()
    def step(self) -> List[RequestOutput]:
        if not self._running_ids: return []
        
        # 1. Schedule: Separate Prefills and Decodes
        prefills = []
        decodes = []
        for rid in self._running_ids:
            if self._requests[rid]["is_prefill"]: prefills.append(rid)
            else: decodes.append(rid)
        
        # SYNC DECODE FAST PATH: When only decodes are present, bypass all Async/Object creation.
        if decodes and not prefills:
            return self._decode_step_sync(decodes)

        results = []
        chunk_size = self._prefill_chunk_size

        # 2. Execute
        # STRATEGY:
        # - Decode-priority mode (default): decode first; reserve prefill budget only when
        #   prefill backlog builds up (or when there is no decode).
        # - Fallback mixed mode: run one prefill chunk + decode micro-batch in each step.
        step_token_budget = max(1, int(self._step_token_budget))
        if self._decode_priority_enabled:
            prefill_token_budget = 0
            if prefills and not decodes:
                prefill_token_budget = min(chunk_size, step_token_budget)
            elif prefills and len(prefills) >= self._prefill_reserve_backlog:
                reserve_tokens = max(
                    self._prefill_reserved_tokens,
                    int(step_token_budget * self._prefill_catchup_ratio),
                )
                prefill_token_budget = min(
                    step_token_budget,
                    max(1, reserve_tokens),
                )
            decode_limit = min(len(decodes), max(0, step_token_budget - prefill_token_budget))
        else:
            if prefills:
                reserve_tokens = max(1, self._prefill_reserved_tokens or 1)
                decode_limit = max(
                    0, min(len(decodes), step_token_budget - reserve_tokens)
                )
            else:
                decode_limit = min(len(decodes), step_token_budget)
            prefill_token_budget = max(0, step_token_budget - decode_limit)
        decodes_to_run = decodes[:decode_limit]

        if prefills and prefill_token_budget > 0:
            # Prefill micro-batch: select requests with the same processed_len so
            # full-attention path can use a consistent kv_chunk_start.
            base_processed_len = self._requests[prefills[0]]["seq_len"]
            candidate_prefills = [
                rid
                for rid in prefills
                if self._requests[rid]["seq_len"] == base_processed_len
            ]
            prefill_batch_size = min(self._prefill_microbatch_size, len(candidate_prefills))
            prefills_to_run = candidate_prefills[:prefill_batch_size]

            # Budget applies to total prefill tokens in this step.
            # chunk_len * batch_size <= prefill_token_budget.
            min_remaining = min(
                len(self._requests[rid]["input_ids"]) - self._requests[rid]["seq_len"]
                for rid in prefills_to_run
            )
            per_req_budget = max(1, prefill_token_budget // max(1, prefill_batch_size))
            this_chunk_len = min(min_remaining, chunk_size, per_req_budget)
            if this_chunk_len <= 0:
                this_chunk_len = 1

            curr_input_rows = []
            position_rows = []
            slot_mapping_rows = []
            block_tables = []
            seq_lens_prefill = []
            kv_start_indices = []
            req_dicts_prefill = [self._requests[rid] for rid in prefills_to_run]
            is_last_chunk_flags = []

            for rid in prefills_to_run:
                req = self._requests[rid]
                slot_idx = req["slot_idx"]
                all_input_ids = req["input_ids"]
                processed_len = req["seq_len"]
                remaining_len = len(all_input_ids) - processed_len
                is_last_chunk = processed_len + this_chunk_len >= len(all_input_ids)
                is_last_chunk_flags.append(is_last_chunk)

                curr_chunk_ids = all_input_ids[
                    processed_len : processed_len + this_chunk_len
                ]
                curr_input_rows.append(curr_chunk_ids)
                position_rows.append(
                    torch.arange(
                        processed_len,
                        processed_len + this_chunk_len,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
                slot_mapping_rows.append(
                    slot_idx * self.max_model_len
                    + torch.arange(
                        processed_len,
                        processed_len + this_chunk_len,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
                start_block = slot_idx * self.num_blocks_per_seq
                block_tables.append(
                    torch.arange(
                        start_block,
                        start_block + self.num_blocks_per_seq,
                        device=self.device,
                        dtype=torch.int32,
                    )
                )
                seq_lens_prefill.append(processed_len + this_chunk_len)
                kv_start_indices.append(processed_len)

            curr_input = torch.tensor(curr_input_rows, device=self.device)
            positions = torch.stack(position_rows, dim=0).to(self.device)
            slot_mapping = torch.cat(slot_mapping_rows, dim=0).to(self.device)
            block_tables_t = torch.stack(block_tables).to(self.device)
            attn_carry_prefill = self._stack_per_layer_carries(
                req_dicts_prefill, self.num_layers, "linear_attn_carry"
            )
            conv_carry_prefill = self._stack_per_layer_carries(
                req_dicts_prefill, self.num_layers, "linear_conv_carry"
            )

            attn_metadata = {
                "slot_mapping": slot_mapping,
                "seq_lens": torch.tensor(
                    seq_lens_prefill, device=self.device, dtype=torch.int32
                ),
                "is_prefill": True,
                "kv_start_indices": torch.tensor(
                    kv_start_indices, device=self.device, dtype=torch.int32
                ),
                "block_tables": block_tables_t,
                "linear_attn_carry": attn_carry_prefill,
                "linear_conv_carry": conv_carry_prefill,
                "kv_scale_cache": self.kv_scale_caches,
                "kv_cache_dtype": self.inf_config.kv_type,
                "k_scale": self.inf_config.k_scale,
                "v_scale": self.inf_config.v_scale,
                "config": self.inf_config,
            }

            try:
                lora_mapping = [self._requests[rid].get("lora_id") for rid in prefills_to_run]
                logits = self.model(
                    curr_input,
                    positions,
                    self.kv_caches,
                    attn_metadata,
                    lora_mapping=lora_mapping,
                )
                self._split_per_layer_carries(
                    attn_metadata["linear_attn_carry"],
                    req_dicts_prefill,
                    "linear_attn_carry",
                )
                self._split_per_layer_carries(
                    attn_metadata["linear_conv_carry"],
                    req_dicts_prefill,
                    "linear_conv_carry",
                )

                for i, rid in enumerate(prefills_to_run):
                    req = self._requests[rid]
                    req["seq_len"] += this_chunk_len
                    if not is_last_chunk_flags[i]:
                        continue
                    # First generated token: respect SamplingParams (temperature, top_p, etc.)
                    eos_mask = _eos_stop_token_ids_for_sampling(
                        self.tokenizer,
                        req["sampling_params"],
                        getattr(self.model_config, "hf_config", None),
                    )
                    token_logits = self.output_processor.apply_context_bias(
                        logits[i, -1, :],
                        req["generated_ids"],
                        req["sampling_params"],
                        req.get("capital_question_bias_token_ids", []),
                        req.get("is_chinese_capital_question", False),
                    )
                    next_token = _sample_token_from_logits(
                        token_logits,
                        req["sampling_params"],
                        req["generated_ids"],
                        req.get("rng"),
                        eos_token_ids_to_mask=eos_mask,
                        anti_template_token_ids=req.get("anti_template_token_ids"),
                    )
                    req["generated_ids"].append(next_token)
                    req["is_prefill"] = False
                    self._process_completion(rid, next_token, results)
            except Exception as e:
                print(f"!!! LiteEngine Error (Chunked Prefill): {e}"); import traceback; traceback.print_exc()
                for rid in prefills_to_run:
                    self._free_request(rid)

        if decodes_to_run:
            # Batch decode micro-step under the per-step token budget.
            input_tokens = []
            slot_indices = []
            seq_lens = []
            pos_indices = []
            
            for rid in decodes_to_run:
                req = self._requests[rid]
                last_token = req["generated_ids"][-1]
                input_tokens.append([last_token])
                slot_indices.append(req["slot_idx"])
                
                current_len = req["seq_len"]
                seq_lens.append(current_len + 1) # Including new token
                pos_indices.append(current_len)
                
            curr_input = torch.tensor(input_tokens, device=self.device) # (B, 1)
            positions = torch.tensor(pos_indices, device=self.device).unsqueeze(1) # (B, 1)
            
            # Generate block tables for batch
            batch_block_tables = []
            for S in slot_indices:
                start_block = S * self.num_blocks_per_seq
                batch_block_tables.append(torch.arange(start_block, start_block + self.num_blocks_per_seq, dtype=torch.int32))
            block_tables = torch.stack(batch_block_tables).to(self.device)

            # slot_mapping maps batch tokens to physical indices
            slot_mapping = torch.tensor([s * self.max_model_len + p for s, p in zip(slot_indices, pos_indices)], device=self.device, dtype=torch.long)

            req_dicts = [self._requests[rid] for rid in decodes_to_run]
            attn_carry_batch = self._stack_per_layer_carries(
                req_dicts, self.num_layers, "linear_attn_carry"
            )
            conv_carry_batch = self._stack_per_layer_carries(
                req_dicts, self.num_layers, "linear_conv_carry"
            )
            attn_metadata = {
                "slot_mapping": slot_mapping,
                "seq_lens": torch.tensor(seq_lens, device=self.device, dtype=torch.int32),
                "block_tables": block_tables,
                "is_prefill": False,
                "kv_start_indices": torch.tensor(
                    pos_indices, device=self.device, dtype=torch.int32
                ),
                "linear_attn_carry": attn_carry_batch,
                "linear_conv_carry": conv_carry_batch,
                "kv_scale_cache": self.kv_scale_caches,
                "kv_cache_dtype": self.inf_config.kv_type,
                "k_scale": self.inf_config.k_scale,
                "v_scale": self.inf_config.v_scale,
                "config": self.inf_config,
            }

            try:
                lora_mapping = [self._requests[rid].get("lora_id") for rid in decodes_to_run]
                logits = self.model(curr_input, positions, self.kv_caches, attn_metadata, lora_mapping=lora_mapping)
                self._split_per_layer_carries(
                    attn_metadata["linear_attn_carry"], req_dicts, "linear_attn_carry"
                )
                self._split_per_layer_carries(
                    attn_metadata["linear_conv_carry"], req_dicts, "linear_conv_carry"
                )
                # logits: (B, 1, Vocab) — per-request sampling (may differ per SamplingParams)
                # OPTIMIZATION: If all requests are greedy (temp=0), we can argmax in one go.
                is_all_greedy = all(self._requests[rid]["sampling_params"].temperature == 0 for rid in decodes_to_run)
                
                if is_all_greedy:
                    # Parallel Argmax
                    next_tokens = torch.argmax(logits[:, -1, :], dim=-1).cpu().tolist()
                    for i, rid in enumerate(decodes_to_run):
                        req_i = self._requests[rid]
                        token = next_tokens[i]
                        req_i["generated_ids"].append(token)
                        req_i["seq_len"] += 1
                        self._process_completion(rid, token, results)
                else:
                    # Standard fallback for mixed sampling params
                    for i, rid in enumerate(decodes_to_run):
                        req_i = self._requests[rid]
                        eos_mask = _eos_stop_token_ids_for_sampling(
                            self.tokenizer,
                            req_i["sampling_params"],
                            getattr(self.model_config, "hf_config", None),
                        )
                        token_logits = self.output_processor.apply_context_bias(
                            logits[i, -1, :],
                            req_i["generated_ids"],
                            req_i["sampling_params"],
                            req_i.get("capital_question_bias_token_ids", []),
                            req_i.get("is_chinese_capital_question", False),
                        )
                        token = _sample_token_from_logits(
                            token_logits,
                            req_i["sampling_params"],
                            req_i["generated_ids"],
                            req_i.get("rng"),
                            eos_token_ids_to_mask=eos_mask,
                            anti_template_token_ids=req_i.get("anti_template_token_ids"),
                        )
                        req_i["generated_ids"].append(token)
                        req_i["seq_len"] += 1
                        self._process_completion(rid, token, results)
                    
            except Exception as e:
                print(f"!!! LiteEngine Error (Decode): {e}"); import traceback; traceback.print_exc()
                # Fail all involved? Or try to isolate? For Lite, just fail.
                for rid in decodes_to_run: self._free_request(rid)

        return results

    def _process_completion(self, rid, next_token, results):
        req = self._requests[rid]
        sp = req["sampling_params"]
        eos_ids = _eos_stop_token_ids_for_sampling(
            self.tokenizer,
            sp,
            getattr(self.model_config, "hf_config", None),
        )
        max_tok = int(sp.max_tokens or 16)
        gen_len = len(req["generated_ids"])
        min_tok = int(getattr(sp, "min_tokens", 0) or 0)

        if getattr(sp, "ignore_eos", False):
            if gen_len >= max_tok:
                req["finished"] = True
        elif next_token in eos_ids and gen_len >= min_tok:
            req["finished"] = True
        elif gen_len >= max_tok:
            req["finished"] = True

        current_text = _decode_generated_text(self.tokenizer, req["generated_ids"], sp)
        if not req["finished"] and self.output_processor.should_early_stop(req['generated_ids'], current_text):
            req["low_info_hits"] = int(req.get("low_info_hits", 0)) + 1
            # Require two consecutive detections to avoid cutting valid short outputs.
            if req["low_info_hits"] >= 2 and gen_len >= max(10, min_tok):
                req["finished"] = True
        else:
            req["low_info_hits"] = 0

        display_text = self.output_processor.cleanup_output_text(current_text)
        completion = CompletionOutput(
            index=0,
            text=display_text,
            token_ids=req["generated_ids"],
            cumulative_logprob=0.0,
        )
        out = RequestOutput(request_id=rid, prompt=req["prompt"], prompt_token_ids=req["input_ids"], outputs=[completion], finished=req["finished"])
        self._request_streams[rid].put_nowait(out)
        results.append(out)
        
        if req["finished"]:
            self._free_request(rid)
