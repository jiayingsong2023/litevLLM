# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list
from vllm.utils.torch_utils import cuda_device_count_stateless

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
EPLBPolicyOption = Literal["default"]
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "allgather_reducescatter",
    "flashinfer_all2allv",
]


@config
@dataclass
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP)."""

    window_size: int = 1000
    """Window size for expert load recording."""
    step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism."""

    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """
    log_balancedness_interval: int = 1
    """
    Interval for logging the balancedness.
    """
    use_async: bool = False
    """
    Whether to use non-blocking EPLB.
    """

    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB)."""

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


@config
@dataclass
class ParallelConfig:
    """Configuration for single-node execution with Data Parallelism."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups (Fixed to 1)."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups (Fixed to 1)."""
    prefill_context_parallel_size: int = 1
    """Number of prefill context parallel groups (Fixed to 1)."""
    data_parallel_size: int = 1
    """Number of data parallel groups for throughput scaling."""
    
    # Compatibility fields for single-node
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    node_rank: int = 0
    nnodes: int = 1
    rank: int = 0
    world_size: int = 1
    
    # Internal fields for compatibility
    data_parallel_rank: int = 0
    data_parallel_index: int = 0
    data_parallel_size_local: int = 1
    data_parallel_rank_local: int = 0
    data_parallel_backend: str = "mp"
    data_parallel_hybrid_lb: bool = False
    data_parallel_external_lb: bool = False
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int = 29500
    data_parallel_rpc_port: int = 29550
    decode_context_parallel_size: int = 1
    dcp_kv_cache_interleave_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE."""
    
    max_parallel_loading_workers: int | None = None
    disable_custom_all_reduce: bool = True
    
    enable_expert_parallel: bool = False
    all2all_backend: str = "naive"
    enable_eplb: bool = False
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    expert_placement_strategy: str = "linear"
    use_sequence_parallel_moe: bool = False
    local_engines_only: bool = True
    use_async_output_proc: bool = False
    
    enable_dbo: bool = False
    ubatch_size: int = 0
    dbo_decode_token_threshold: int = 32
    dbo_prefill_token_threshold: int = 512
    disable_nccl_for_dp_synchronization: bool = True
    
    _api_process_count: int = 1
    _api_process_rank: int = 0
    
    ray_workers_use_nsight: bool = False
    ray_runtime_env: Any = None
    placement_group: Any = None
    
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"
    worker_extension_cls: str = ""
    distributed_executor_backend: str = "uni"

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        # Enforce single node constraints
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.prefill_context_parallel_size = 1
        self.nnodes = 1
        self.distributed_executor_backend = "uni"
        self.world_size = self.data_parallel_size
        return self

    @property
    def world_size_across_dp(self) -> int:
        return self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else self.ubatch_size

    def compute_hash(self):
        from vllm.config.utils import get_hash_factors, hash_factors
        factors = get_hash_factors(self, set())
        return hash_factors(factors)

    def __post_init__(self) -> None:
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.world_size = self.data_parallel_size
        self.distributed_executor_backend = "uni"

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        return self

    def replace(self, **kwargs) -> Self:
        return replace(self, **kwargs)
