# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Qwen2 model optimized for LiteEngine and Triton."""

import torch
from torch import nn
from typing import Iterable, Optional, Set, Tuple, Any
from transformers import Qwen2Config

from vllm.config import VllmConfig
from vllm.model_executor.models.lite_base import LiteForCausalLM, LiteModel

# Use generic implementations directly
Qwen2Model = LiteModel
Qwen2ForCausalLM = LiteForCausalLM