# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class for litevLLM (Single process, Triton only)."""

import gc
import os
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class Worker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.profiler: Any | None = None
        
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "CUDA"],
            )
        elif profiler_config.profiler == "cuda":
            self.profiler = CudaProfilerWrapper(profiler_config)

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        # Forced to use cuda:0 for single card
        self.device = torch.device("cuda:0")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Simplification: No distributed environment to initialize
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance
        init_batch_invariance(self.vllm_config.attention_config.backend)

        set_random_seed(self.model_config.seed)
        gc.collect()
        torch.cuda.empty_cache()

        self.init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(self.init_snapshot, self.cache_config)
        
        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(self.vllm_config, self.device) # type: ignore
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        if self.rank == 0:
            report_usage_stats(self.vllm_config)

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(eep_scale_up=False)

    def determine_available_memory(self) -> int:
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )
        return int(self.available_kv_cache_memory_bytes)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        warmup_sizes = []
        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        for size in sorted(warmup_sizes, reverse=True):
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Warm up sampler
        max_num_reqs = min(
            self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_batched_tokens,
        )
        hidden_states, last_hidden_states = self.model_runner._dummy_run(
            num_tokens=max_num_reqs,
            skip_eplb=True,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        if self.model_runner.is_pooling_model:
            self.model_runner._dummy_pooler_run(hidden_states)
        else:
            self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        set_random_seed(self.model_config.seed)

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output, None)

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def shutdown(self) -> None:
        if self.profiler is not None:
            self.profiler.shutdown()

    # Stub methods for compatibility
    def add_lora(self, lora_request: LoRARequest) -> bool: return self.model_runner.add_lora(lora_request)
    def remove_lora(self, lora_id: int) -> bool: return self.model_runner.remove_lora(lora_id)
    def list_loras(self) -> set[int]: return self.model_runner.list_loras()
    def pin_lora(self, lora_id: int) -> bool: return self.model_runner.pin_lora(lora_id)
    def get_model(self) -> nn.Module: return self.model_runner.get_model()
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: return self.model_runner.get_supported_tasks()
    def check_health(self) -> None: pass
