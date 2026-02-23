# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Llama model optimized for LiteEngine and Triton."""

import torch
from torch import nn
from typing import Iterable, Optional, Tuple, Any
from transformers import LlamaConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.lite_base import LiteForCausalLM, LiteModel, LiteDecoderLayer

# Use generic implementations directly
LlamaModel = LiteModel
LlamaForCausalLM = LiteForCausalLM
LlamaDecoderLayer = LiteDecoderLayer