# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.structured_output.backend_guidance import GuidanceBackend
from vllm.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)

class StructuredOutputManager:
