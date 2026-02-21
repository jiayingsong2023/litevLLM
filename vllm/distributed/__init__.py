# SPDX-License-Identifier: Apache-2.0
"""Mock parallel state for single-node execution."""

import torch
from contextlib import contextmanager

class MockGroup:
    def __init__(self):
        self.rank = 0
        self.rank_in_group = 0
        self.world_size = 1
        self.device_group = None
        self.cpu_group = None
        self.is_first_rank = True
        self.is_last_rank = True

    def get_group(self): return self

_MOCK_GROUP = MockGroup()

def get_tp_group(): return _MOCK_GROUP
def get_pp_group(): return _MOCK_GROUP
def get_dp_group(): return _MOCK_GROUP
def get_ep_group(): return _MOCK_GROUP
def get_pcp_group(): return _MOCK_GROUP
def get_dcp_group(): return _MOCK_GROUP

def get_tensor_model_parallel_world_size(): return 1
def get_tensor_model_parallel_rank(): return 0
def get_pipeline_model_parallel_world_size(): return 1
def get_pipeline_model_parallel_rank(): return 0

def tensor_model_parallel_all_gather(input_): return input_
def tensor_model_parallel_gather(input_, dst=0, dim=-1): return input_
def tensor_model_parallel_all_reduce(input_): return input_
def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    return [tensor]
def divide(x, y): return x // y

def ensure_model_parallel_initialized(*args, **kwargs): pass
def init_distributed_environment(*args, **kwargs): pass
def set_custom_all_reduce(*args, **kwargs): pass
def stateless_destroy_torch_distributed_process_group(*args, **kwargs): pass

class GroupCoordinator:
    def __init__(self, *args, **kwargs):
        self.rank = 0
        self.world_size = 1
        self.device_group = None
        self.cpu_group = None

def is_local_first_rank(): return True
def is_global_first_rank(): return True

def has_kv_transfer_group(): return False
def has_ec_transfer_group(): return False
def get_world_size(): return 1
def get_rank(): return 0

@contextmanager
def graph_capture():
    yield

def prepare_communication_buffer_for_model(*args, **kwargs):
    pass

def get_pp_indices(num_layers, world_size, rank):
    return 0, num_layers