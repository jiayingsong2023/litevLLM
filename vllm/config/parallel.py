# SPDX-License-Identifier: Apache-2.0
"""
Simplified ParallelConfig for single-GPU LiteEngine.
Forces tensor_parallel_size=1 and pipeline_parallel_size=1.
"""

from dataclasses import replace
from typing import Any, Self, Literal
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]

@config
@dataclass
class EPLBConfig:
    """Mock EPLBConfig for compatibility with MoE models."""
    num_redundant_experts: int = 0
    enable_eplb: bool = False

@config
@dataclass
class ParallelConfig:
    """Configuration for single-GPU execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups (Fixed to 1)."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups (Fixed to 1)."""
    prefill_context_parallel_size: int = 1
    """Fixed to 1 for LiteEngine."""
    data_parallel_size: int = 1
    """Fixed to 1 for single-GPU inference."""
    
    # Compatibility fields for vLLM core
    rank: int = 0
    world_size: int = 1
    node_rank: int = 0
    nnodes: int = 1
    
    # DP Compatibility
    data_parallel_rank: int = 0
    data_parallel_rank_local: int = 0
    data_parallel_size_local: int = 1
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29500
    data_parallel_backend: str = "mp"
    
    # CP Compatibility
    decode_context_parallel_size: int = 1
    dcp_kv_cache_interleave_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    
    # Distributed environment stubs
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    
    # Internal flags
    distributed_executor_backend: str = "uni"
    worker_cls: str = "auto"
    worker_extension_cls: str = ""
    
    # API stubs
    _api_process_count: int = 1
    _api_process_rank: int = 0
    local_engines_only: bool = True
    
    # MoE compatibility
    is_moe_model: bool | None = None
    enable_expert_parallel: bool = False
    all2all_backend: str = "naive"
    enable_eplb: bool = False
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    use_sequence_parallel_moe: bool = False
    expert_placement_strategy: str = "linear"
    
    # Performance stubs
    disable_custom_all_reduce: bool = True
    max_parallel_loading_workers: int | None = None
    
    # DBO Compatibility
    enable_dbo: bool = False
    ubatch_size: int = 0
    disable_nccl_for_dp_synchronization: bool = True

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        # Enforce single-GPU constraints regardless of user input
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.prefill_context_parallel_size = 1
        self.data_parallel_size = 1
        self.nnodes = 1
        self.rank = 0
        self.world_size = 1
        self.distributed_executor_backend = "uni"
        return self

    @property
    def data_parallel_index(self) -> int:
        return self.data_parallel_rank

    @property
    def world_size_across_dp(self) -> int:
        return 1

    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else (self.ubatch_size if self.ubatch_size > 0 else 1)

    def replace(self, **kwargs) -> Self:
        return replace(self, **kwargs)