# SPDX-License-Identifier: Apache-2.0
import torch
from .parallel_state import *

def divide(numerator, denominator):
    return numerator // denominator

def split_tensor_along_last_dim(tensor, num_partitions):
    return torch.split(tensor, tensor.shape[-1] // num_partitions, dim=-1)

def get_pcp_group(): return get_pp_group()
def get_dcp_group(): return get_pp_group()

def tensor_model_parallel_all_reduce(input_): return input_
def tensor_model_parallel_all_gather(input_, dim=-1): return input_
def tensor_model_parallel_gather(input_, dst=0, dim=-1): return input_
def tensor_model_parallel_broadcast(input_, src=0): return input_

def broadcast_tensor_dict(data=None, src=0, group=None): return data

def cleanup_dist_env_and_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def stateless_destroy_torch_distributed_process_group(): pass

__all__ = [
    "divide", "split_tensor_along_last_dim", "get_tensor_model_parallel_world_size",
    "get_pipeline_model_parallel_world_size", "get_tensor_model_parallel_rank",
    "get_pipeline_model_parallel_rank", "get_pcp_group", "get_dcp_group",
    "is_global_first_rank", "is_local_first_rank", "init_distributed_environment",
    "initialize_model_parallel", "model_parallel_is_initialized", "get_pp_indices",
    "tensor_model_parallel_all_reduce", "tensor_model_parallel_all_gather",
    "tensor_model_parallel_gather", "tensor_model_parallel_broadcast",
    "broadcast_tensor_dict", "cleanup_dist_env_and_memory",
    "stateless_destroy_torch_distributed_process_group",
]
