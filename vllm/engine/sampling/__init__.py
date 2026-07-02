# SPDX-License-Identifier: Apache-2.0
from vllm.engine.sampling.utils import (
    eos_stop_token_ids_for_sampling,
    hf_config_eos_token_ids,
)

__all__ = ["eos_stop_token_ids_for_sampling", "hf_config_eos_token_ids"]
