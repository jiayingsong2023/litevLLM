# SPDX-License-Identifier: Apache-2.0
from .parallel_state import *

def divide(a, b): return a // b

def get_dcp_group(): return None
def get_pcp_group(): return None

def cleanup_dist_env_and_memory(): pass

# Communication Ops Stubs
def tensor_model_parallel_all_reduce(input_): return input_
def tensor_model_parallel_all_gather(input_): return input_
