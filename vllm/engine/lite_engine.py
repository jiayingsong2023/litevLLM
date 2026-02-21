# SPDX-License-Identifier: Apache-2.0
"""
LiteEngine: Optimized, Flattened, Single-GPU Inference Engine.
Standardizes on Triton kernels and removes all distributed overhead.
"""

import asyncio
import gc
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, AsyncEngineArgs
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.core.sched.scheduler import Scheduler
from vllm.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.platforms import current_platform
from vllm.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.structured_output import StructuredOutputManager
from vllm.sample.sampler import Sampler
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

class LiteEngine:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        
        # 1. Force Triton Path
        self.vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
        
        # 2. Setup Device
        self.device = torch.device("cuda:0")
        current_platform.set_device(self.device)
        
        # 3. Direct Model Load
        logger.info(f"LiteEngine: Loading model {self.model_config.model}...")
        self.model = get_model(vllm_config=self.vllm_config)
        
        # 4. Profile & Allocate KV Cache
        self._initialize_engine()
        
        # 5. Setup Infrastructure
        self.structured_output_manager = StructuredOutputManager(self.vllm_config)
        self.sampler = Sampler()
        
        # 6. Initialize Scheduler
        kv_cache_config = generate_scheduler_kv_cache_config(self.vllm_config)
        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            block_size=self.cache_config.block_size,
            log_stats=True
        )
        
        self._request_streams: Dict[str, asyncio.Queue] = {}
        logger.info("LiteEngine: Optimized single-GPU engine ready.")

    def _initialize_engine(self):
        """Profile memory and allocate KV cache blocks."""
        logger.info("LiteEngine: Profiling memory usage...")
        
        # 1. Profile: Run a dummy forward pass to get peak memory usage
        # This is a simplified version of ModelRunner.profile_run
        with torch.no_grad():
            # Create dummy inputs for profiling
            dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            dummy_positions = torch.zeros((1,), dtype=torch.long, device=self.device)
            
            # Temporary dummy cache for profiling
            # (In a real engine, we'd calculate exact max_num_blocks here)
            start_mem = torch.cuda.memory_allocated(self.device)
            self.model(input_ids=dummy_input_ids, positions=dummy_positions, kv_caches=None, attn_metadata=None)
            end_mem = torch.cuda.memory_allocated(self.device)
            model_mem = end_mem - start_mem
            logger.info(f"LiteEngine: Model weight/act memory: {model_mem / 1024**3:.2f} GB")

        # 2. Allocate KV Cache based on remaining memory
        # For now, we use a stable default or config value
        if self.cache_config.num_gpu_blocks is None:
            self.cache_config.num_gpu_blocks = 2048 
            
        from vllm.attention.selector import get_attn_backend
        backend_cls = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.cache_config.block_size,
        )
        self.attn_backend = backend_cls(self.vllm_config)
        self.kv_cache = self.attn_backend.get_impl_cls().allocate_kv_cache(
            num_blocks=self.cache_config.num_gpu_blocks,
            device=self.device,
        )
        logger.info(f"LiteEngine: Allocated {self.cache_config.num_gpu_blocks} blocks of KV Cache.")

    def _prepare_inputs(self, scheduler_output: Any):
        """Prepare flattened tensors and Triton-specific metadata."""
        input_ids_list = []
        positions_list = []
        slot_mapping_list = []
        
        for req in scheduler_output.scheduled_requests:
            input_ids_list.append(torch.tensor(req.token_ids, device=self.device))
            positions_list.append(torch.tensor(req.positions, device=self.device))
            # Slot mapping logic for PagedAttention
            # req.slots maps tokens to physical offsets in self.kv_cache
            slot_mapping_list.append(torch.tensor(req.slots, device=self.device, dtype=torch.long))
            
        input_ids = torch.cat(input_ids_list)
        positions = torch.cat(positions_list)
        slot_mapping = torch.cat(slot_mapping_list)
        
        # Build Triton-specific attention metadata
        from vllm.attention.backends.triton_attn import TritonAttentionMetadata
        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=len(input_ids),
            max_query_len=scheduler_output.max_query_len,
            query_start_loc=torch.tensor(scheduler_output.query_start_loc, device=self.device),
            max_seq_len=scheduler_output.max_seq_len,
            seq_lens=torch.tensor(scheduler_output.seq_lens, device=self.device),
            block_table=torch.tensor(scheduler_output.block_table, device=self.device),
            slot_mapping=slot_mapping,
            # Triton specific tuning parameters
            seq_threshold_3D=128,
            num_par_softmax_segments=16,
            softmax_segm_output=None,
            softmax_segm_max=None,
            softmax_segm_expsum=None,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None
        )
        
        return input_ids, positions, attn_metadata

    async def step(self) -> List[RequestOutput]:
        """Core inference step integrating Continuous Batching."""
        # 1. Scheduler: Get next batch of tokens to process
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.scheduled_requests:
            return []

        # 2. Input Prep: Direct-to-Triton Tensor mapping
        input_ids, positions, attn_metadata = self._prepare_inputs(scheduler_output)

        # 3. Model Forward: Pure Triton compute path
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_cache,
            attn_metadata=attn_metadata
        )

        # 4. Sampling: Fast single-GPU sampler
        # We reuse vllm sampler but with optimized metadata
        from vllm.sample.metadata import SamplingMetadata
        sampling_metadata = SamplingMetadata.from_scheduler_output(scheduler_output, self.device)
        sampler_output = self.sampler(logits, sampling_metadata)

        # 5. Update: Notify streams and update scheduler state
        outputs = self.scheduler.update_from_output(sampler_output, scheduler_output)
        
        for output in outputs:
            if output.request_id in self._request_streams:
                self._request_streams[output.request_id].put_nowait(output)
                
        return outputs