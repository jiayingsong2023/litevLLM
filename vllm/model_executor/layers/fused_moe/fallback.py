# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

class FallbackExperts(mk.FusedMoEPermuteExpertsUnpermute, ABC):
        Get the cls for the experts and fallback experts.

        Subclasses should implement this method, so that
        we have a consistent way to call the _supports_*
        class methods below.
