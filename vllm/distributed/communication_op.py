# SPDX-License-Identifier: Apache-2.0
"""LitevLLM Communication Ops: Identity operations for single GPU."""

def tensor_model_parallel_all_reduce(input_):
    return input_

def tensor_model_parallel_all_gather(input_, dim=-1):
    return input_

def tensor_model_parallel_gather(input_, dst=0, dim=-1):
    return input_

def tensor_model_parallel_broadcast(input_, src=0):
    return input_
