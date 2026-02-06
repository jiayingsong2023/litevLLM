# SPDX-License-Identifier: Apache-2.0

"""litevLLM - Simplified distributed state (Single process, single card only)."""

import torch
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, List, Dict
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorStub,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

class GroupCoordinator:
    """Mock GroupCoordinator for single process."""
    def __init__(self, group_name: str = "anonymous"):
        self.unique_name = group_name
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.rank_in_group = 0
        self.ranks = [0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_communicator = DeviceCommunicatorStub()

    @property
    def first_rank(self): return 0
    @property
    def last_rank(self): return 0
    @property
    def is_first_rank(self): return True
    @property
    def is_last_rank(self): return True

    def barrier(self): pass
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor: return input_
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: return input_
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: return input_
    def broadcast(self, input_: torch.Tensor, src: int = 0): return input_
    def broadcast_object(self, obj: Any = None, src: int = 0): return obj
    def destroy(self): pass
    def prepare_communication_buffer_for_model(self, model: torch.nn.Module): pass

# Global instances for litevLLM
_SINGLE_COORD = GroupCoordinator("lite_coord")

def get_world_group(): return _SINGLE_COORD
def get_tp_group(): return _SINGLE_COORD
def get_pp_group(): return _SINGLE_COORD
def get_dp_group(): return _SINGLE_COORD
def get_ep_group(): return _SINGLE_COORD
def get_dcp_group(): return _SINGLE_COORD
def get_pcp_group(): return _SINGLE_COORD
def get_tensor_model_parallel_group(): return _SINGLE_COORD

def get_tensor_model_parallel_world_size(): return 1
def get_tensor_model_parallel_rank(): return 0
def get_pipeline_model_parallel_world_size(): return 1
def get_pipeline_model_parallel_rank(): return 0
def get_node_count(): return 1

def init_distributed_environment(*args, **kwargs):
    logger.info("litevLLM: Distributed environment bypassed (single process mode).")

def initialize_model_parallel(*args, **kwargs):
    logger.info("litevLLM: Model parallel initialization bypassed.")

def prepare_communication_buffer_for_model(model: torch.nn.Module) -> None:
    pass

def set_custom_all_reduce(enabled: bool) -> None:
    pass

def model_parallel_is_initialized(): return True
def ensure_model_parallel_initialized(*args, **kwargs): pass
def destroy_model_parallel(): pass
def destroy_distributed_environment(): pass
def stateless_destroy_torch_distributed_process_group(pg: Any) -> None: pass
def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    if shutdown_ray:
        import ray
        ray.shutdown()

def is_global_first_rank(): return True
def is_local_first_rank(): return True

def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return tensor

def tensor_model_parallel_all_gather(
    tensor: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return tensor

def tensor_model_parallel_gather(
    tensor: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return tensor

def tensor_model_parallel_reduce_scatter(
    tensor: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return tensor

@contextmanager
def graph_capture(device: torch.device):
    from vllm.distributed.parallel_state import GraphCaptureContext
    yield GraphCaptureContext(torch.cuda.Stream(device=device))

class GraphCaptureContext:
    def __init__(self, stream):
        self.stream = stream
