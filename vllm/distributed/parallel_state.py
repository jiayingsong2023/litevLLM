# SPDX-License-Identifier: Apache-2.0
"""LitevLLM: Simplified parallel state for single-GPU inference."""

from typing import Any, Optional
from contextlib import contextmanager

class GroupCoordinator:
    """Stub GroupCoordinator for single GPU."""
    def __init__(self, *args, **kwargs):
        self.world_size = 1
        self.rank = 0
        self.rank_in_group = 0
        self.is_first_rank = True
        self.is_last_rank = True
        self.device_group = None
        self.cpu_group = None

    def barrier(self): pass
    def destroy(self): pass

# Constants
_TP_COORDINATOR: Optional[GroupCoordinator] = None
_PP_COORDINATOR: Optional[GroupCoordinator] = None

def get_tp_group(): return GroupCoordinator()
def get_pp_group(): return GroupCoordinator()
def get_ep_group(): return GroupCoordinator()
def get_dp_group(): return GroupCoordinator()
def get_world_group(): return GroupCoordinator()
def get_pcp_group(): return GroupCoordinator()
def get_dcp_group(): return GroupCoordinator()

def get_tensor_model_parallel_world_size(): return 1
def get_pipeline_model_parallel_world_size(): return 1
def get_tensor_model_parallel_rank(): return 0
def get_pipeline_model_parallel_rank(): return 0

def is_global_first_rank(): return True
def is_local_first_rank(): return True

def has_ec_transfer_group(): return False
def has_kv_transfer_group(): return False

def init_distributed_environment(*args, **kwargs):
    """No-op initialization for LitevLLM."""
    pass

def initialize_model_parallel(*args, **kwargs): pass
def model_parallel_is_initialized(): return True

def get_pp_indices(num_layers, pp_rank, pp_size):
    return 0, num_layers

@contextmanager
def graph_capture():
    """Stub graph_capture context manager."""
    yield

def prepare_communication_buffer_for_model(*args, **kwargs):
    """No-op for single GPU."""
    pass
