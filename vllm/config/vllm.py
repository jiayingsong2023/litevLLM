# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import torch
# from pydantic import ConfigDict, Field, model_validator
# from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.utils import random_uuid
from vllm.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .load import LoadConfig
from .lora import LoRAConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .parallel import ParallelConfig
from .scheduler import SchedulerConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
else:
    PretrainedConfig = Any
    QuantizationConfig = Any

logger = init_logger(__name__)

class OptimizationLevel(IntEnum):
    """Optimization level enum."""
    O0 = 0
    O1 = 1
    O2 = 2
    O3 = 3

IS_QUANTIZED = False
IS_DENSE = False

def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    return cfg.compilation_config.is_custom_op_enabled("rms_norm") or cfg.compilation_config.is_custom_op_enabled("quant_fp8")

def enable_act_fusion(cfg: "VllmConfig") -> bool:
    return cfg.compilation_config.is_custom_op_enabled("silu_and_mul") or cfg.compilation_config.is_custom_op_enabled("quant_fp8")

OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": False,
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": False,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_02,
}

@config
@dataclass
class VllmConfig:
    """LitevLLM Config using standard dataclass."""

    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig.default_factory)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    load_config: LoadConfig = field(default_factory=LoadConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    lora_config: LoRAConfig | None = None
    structured_outputs_config: StructuredOutputsConfig = field(default_factory=StructuredOutputsConfig)
    observability_config: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    quant_config: QuantizationConfig | None = None
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    
    # Compatibility stubs
    speculative_config: Any = None
    kv_transfer_config: Any = None
    kv_events_config: Any = None
    ec_transfer_config: Any = None
    profiler_config: Any = None
    
    additional_config: dict = field(default_factory=dict)
    instance_id: str = ""
    optimization_level: OptimizationLevel = OptimizationLevel.O2

    def enable_trace_function_call_for_thread(self):
        pass

    def compile_debug_dump_path(self) -> Path | None:
        if self.compilation_config.debug_dump_path is None:
            return None
        return self.compilation_config.debug_dump_path / "rank_0_dp_0"

    def compute_hash(self) -> str:
        factors = [self.model_config.compute_hash() if self.model_config else "None"]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

    @property
    def needs_dp_coordinator(self) -> bool:
        return False

    @staticmethod
    def _get_quantization_config(model_config: ModelConfig, load_config: LoadConfig) -> QuantizationConfig | None:
        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import get_quant_config
            return get_quant_config(model_config, load_config)
        return None

    def __post_init__(self):
        from vllm.config.compilation import CompilationMode, CUDAGraphMode
        self.instance_id = f"{time.time_ns()}"
        if self.model_config is not None:
            self.parallel_config.is_moe_model = self.model_config.is_moe
        
        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(self.model_config, self.load_config)

        # Force async scheduling for LiteEngine if supported
        if self.scheduler_config.async_scheduling is None:
            self.scheduler_config.async_scheduling = True

        if self.compilation_config.mode is None:
            self.compilation_config.mode = CompilationMode.VLLM_COMPILE if self.optimization_level > OptimizationLevel.O0 else CompilationMode.NONE

        if self.model_config is not None and self.model_config.enforce_eager:
            self.optimization_level = OptimizationLevel.O0
            self.compilation_config.mode = CompilationMode.NONE

        # Initialize cudagraph_mode if not set
        if self.compilation_config.cudagraph_mode is None:
            if self.compilation_config.mode == CompilationMode.NONE:
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            else:
                self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE

        # Ensure custom_ops has a base policy
        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if self.compilation_config.backend == "inductor" and self.compilation_config.mode != CompilationMode.NONE:
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        from vllm.platforms import current_platform
        current_platform.check_and_update_config(self)

        # Initialize splitting_ops for v1
        self.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=self.parallel_config.all2all_backend,
            data_parallel_size=1,
        )

        self._set_compile_ranges()
        if current_platform.support_static_graph_mode():
            self._set_cudagraph_sizes()

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)

    def _set_compile_ranges(self):
        compilation_config = self.compilation_config
        computed_compile_ranges_split_points = []
        compile_range_end = self.scheduler_config.max_num_batched_tokens
        if compile_range_end is not None:
            computed_compile_ranges_split_points.append(compile_range_end)
        
        if compilation_config.compile_ranges_split_points is not None:
            for x in compilation_config.compile_ranges_split_points:
                if compile_range_end is not None and x < compile_range_end and x > 1:
                    computed_compile_ranges_split_points.append(x)
        compilation_config.compile_ranges_split_points = sorted(computed_compile_ranges_split_points)

    def _set_cudagraph_sizes(self):
        if self.model_config is not None and not self.model_config.enforce_eager and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            max_cudagraph_capture_size = self.compilation_config.max_cudagraph_capture_size
            if max_cudagraph_capture_size is None:
                max_cudagraph_capture_size = min(self.scheduler_config.max_num_seqs * 2, 512)
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)
            
            if self.compilation_config.cudagraph_capture_sizes is not None:
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                cudagraph_capture_sizes = [i for i in dedup_sizes if i <= max_num_tokens]
                cudagraph_capture_sizes.sort()
            else:
                cudagraph_capture_sizes = [i for i in [1, 2, 4] if i <= max_cudagraph_capture_size]
                if max_cudagraph_capture_size >= 8:
                    cudagraph_capture_sizes += list(range(8, min(max_cudagraph_capture_size + 1, 256), 8))
                if max_cudagraph_capture_size >= 256:
                    cudagraph_capture_sizes += list(range(256, max_cudagraph_capture_size + 1, 16))
            
            valid_max_size = cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            self.compilation_config.max_cudagraph_capture_size = valid_max_size
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []
        self.compilation_config.post_init_cudagraph_sizes()

    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key): continue
                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                elif current is None:
                    setattr(config_obj, key, value(self) if callable(value) else value)
        apply_recursive(self, defaults)

    def __str__(self):
        return f"VllmConfig(model={self.model_config.model if self.model_config else 'None'})"

_current_vllm_config: VllmConfig | None = None

@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig, check_compile=False, prefix: str | None = None):
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    _current_vllm_config = vllm_config
    try:
        yield
    finally:
        _current_vllm_config = old_vllm_config

@lru_cache(maxsize=1)
def get_cached_compilation_config():
    return get_current_vllm_config().compilation_config

def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError("Current vLLM config is not set.")
    return _current_vllm_config

def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config

T = TypeVar("T")

def get_layers_from_vllm_config(
    vllm_config: VllmConfig,
    layer_type: type[T],
    layer_names: list[str] | None = None,
) -> dict[str, T]:
    if layer_names is None:
        layer_names = list(vllm_config.compilation_config.static_forward_context.keys())
    forward_context = vllm_config.compilation_config.static_forward_context
    return {
        layer_name: forward_context[layer_name]
        for layer_name in layer_names
        if isinstance(forward_context[layer_name], layer_type)
    }