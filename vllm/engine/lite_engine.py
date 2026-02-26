# SPDX-License-Identifier: Apache-2.0

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
