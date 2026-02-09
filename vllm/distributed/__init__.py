# SPDX-License-Identifier: Apache-2.0
"""Mock parallel state for single-node execution."""

from vllm.distributed.parallel_state import (
    get_tp_group,
    get_pp_group,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
    tensor_model_parallel_all_reduce,
    split_tensor_along_last_dim,
    divide,
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    stateless_destroy_torch_distributed_process_group,
    GroupCoordinator,
    is_local_first_rank,
    is_global_first_rank,
    graph_capture,
    prepare_communication_buffer_for_model,
)

