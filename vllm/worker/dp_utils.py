# SPDX-License-Identifier: Apache-2.0

from vllm.config import CUDAGraphMode

def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    parallel_config,
    allow_microbatching: bool,
    allow_dp_padding: bool,
    num_tokens_padded: int | None = None,
    uniform_decode=None,
    num_scheduled_tokens_per_request=None,
    cudagraph_mode: int | None = None,
):
