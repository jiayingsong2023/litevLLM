# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import inspect
import itertools
from abc import abstractmethod
from collections.abc import Sequence
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import guard_cuda_initialization
from vllm.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
    process_dict_updates,
)
from vllm.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm.sample.logits_processor.state import BatchUpdateBuilder, LogitsProcessors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Error message when the user tries to initialize vLLM with a pooling model
# and custom logitsproces
STR_POOLING_REJECTS_LOGITSPROCS = (
    "Pooling models do not support custom logits processors."
)

# Error message when the user tries to initialize vLLM with a speculative
# decoding enabled and custom logitsproces
STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)

LOGITSPROCS_GROUP = "vllm.logits_processors"

BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]

def _load_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    names (FQCNs).

    Effectively, a mixed list of logitproc types and FQCN strings is converted
    into a list of entirely logitproc types, by loading from the FQCNs.

    FQCN syntax is <module>:<type> i.e. x.y.z:CustomLogitProc

    Already-loaded logitproc types must be subclasses of LogitsProcessor

    Args:
      logits_processors: Potentially mixed list of logitsprocs types and FQCN
                         strings for logitproc types

    Returns:
      List of logitproc types

    * First load all installed logitproc plugins
    * Second load custom logitsprocs pass by the user at initialization time

    Args:
      logits_processors: potentially mixed list of logitproc types and
                         logitproc type fully-qualified names (FQCNs)
                         which need to be loaded

    Returns:
      A list of all loaded logitproc types

    To wrap a specific per-request logits processor,
    * Subclass `AdapterLogitsProcessor`
    * Implement `self.is_argmax_invariant()` base-class method
    * Implement `self.new_req_logits_processor(params)`

    `self.__init__(vllm_config, device, is_pin_memory)` does not need to be
    overridden in general. However, to implement custom constructor behavior -
    especially any logic which operates on or stores `vllm_config`, `device`,
    or `is_pin_memory` - `self.__init__(vllm_config, device, is_pin_memory)`
    must be overridden and the override must call
    `super().__init__(vllm_config, device, is_pin_memory)`
        `super().__init__(vllm_config, device, is_pin_memory)`.

        Subclass constructor may find it useful to utilize the `vllm_config`,
        `device` and `is_pin_memory` argument. However regardless of whether
        these arguments are used, the vLLM logits processor interface requires
        all three arguments to be present.

        Return None if logits processor does not need to be applied to request

        Args:
          params: request sampling params

        Returns:
          None if logits processor should not be applied to request; otherwise
          returns a `RequestLogitsProcessor` instance

        Returns None if logits processor is not applicable to request

        Args:
          params: request sampling params
          prompt_ids: request prompt token ids
          output_ids: decoded tokens so far for this request

        Returns:
          logits processor partial[Tensor] or None

