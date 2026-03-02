# SPDX-License-Identifier: Apache-2.0
"""
LitevLLM: Simplified Parallel State for Single-GPU Execution.
This module replaces the complex distributed state management with simple stubs.
"""
from contextlib import contextmanager

def get_tp_group(): return None
def get_pp_group(): return None
def get_dp_group(): return None
def get_ep_group(): return None
def get_world_group(): return None

def get_tensor_model_parallel_world_size(): return 1
def get_pipeline_model_parallel_world_size(): return 1
def get_tensor_model_parallel_rank(): return 0
def get_pipeline_model_parallel_rank(): return 0

def model_parallel_is_initialized(): return True
def is_global_first_rank(): return True

@contextmanager
def graph_capture():
    yield

def prepare_communication_buffer_for_model(*args, **kwargs):
    pass

def init_distributed_environment(*args, **kwargs):
    pass

def initialize_model_parallel(*args, **kwargs):
    pass
