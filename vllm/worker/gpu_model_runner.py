# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.attention.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

class GPUModelRunner:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.device = device
        self.dtype = self.model_config.dtype
        
        # Core model instance
        self.model: Optional[nn.Module] = None
        
        # Pre-allocated buffers for performance (Lite version)
        self.max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.input_ids = torch.empty(self.max_tokens, dtype=torch.int32, device=self.device)
        self.positions = torch.empty(self.max_tokens, dtype=torch.int64, device=self.device)

    def load_model(self):
        """Load model and weights for single-GPU inference."""
        self.model = get_model(vllm_config=self.vllm_config)
        logger.info("LiteModelRunner: Model loaded successfully.")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: Any,
        kv_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Main execution entry point.
        1. Prepare flattened inputs (input_ids, positions).
        2. Build AttentionMetadata.
        3. Forward pass.
        """
        # 1. Prepare Inputs
        input_ids, positions, attn_metadata = self._prepare_inputs(scheduler_output)
        
        # 2. Forward Pass
        # We pass directly to the model which now uses TritonAttention
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata
        )
        
        return logits

    def _prepare_inputs(self, scheduler_output: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """LitevLLM: Simplified input preparation for single GPU."""
        # Flattening sequence tokens into a single batch
        input_ids_list = []
        positions_list = []
        slot_mapping_list = []
        
        for req in scheduler_output.scheduled_seq_groups:
            # Assuming LitevLLM simplified sequence group structure
            input_ids_list.append(req.get_input_ids())
            positions_list.append(req.get_positions())
            slot_mapping_list.append(req.get_slot_mapping())
            
        input_ids = torch.cat(input_ids_list).to(self.device)
        positions = torch.cat(positions_list).to(self.device)
        
        # Mock Attention Metadata for Triton
        attn_metadata = {
            "slot_mapping": torch.cat(slot_mapping_list).to(self.device),
            "seq_lens": scheduler_output.seq_lens,
            "num_tokens": len(input_ids)
        }
        
        return input_ids, positions, attn_metadata