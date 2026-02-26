# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import dataclasses
import functools
import json
import sys
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from itertools import permutations
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import huggingface_hub
import regex as re
import torch
from pydantic import Field, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    ConfigType,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    ParallelConfig,
    PoolerConfig,
    SchedulerConfig,
    StructuredOutputsConfig,
    VllmConfig,
    get_attr_docs,
)
from vllm.config.cache import (
    BlockSize,
    CacheDType,
    KVOffloadingBackend,
    MambaCacheMode,
    MambaDType,
    PrefixCachingHashAlgo,
)
from vllm.config.device import Device
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    LogprobsMode,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode
from vllm.config.observability import DetailedTraceModules
from vllm.config.scheduler import SchedulerPolicy
from vllm.config.utils import get_field
from vllm.config.vllm import OptimizationLevel
from vllm.logger import init_logger, suppress_logging
from vllm.platforms import CpuArchEnum, current_platform
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.config import (
    is_interleaved,
    maybe_override_with_speculators,
)
from vllm.transformers_utils.gguf_utils import is_gguf
from vllm.transformers_utils.repo_utils import get_model_path
from vllm.transformers_utils.utils import is_cloud_storage
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.model_loader import LoadFormats
    from vllm.usage.usage_lib import UsageContext
    from vllm.executor import Executor
else:
    Executor = Any
    QuantizationMethods = Any
    LoadFormats = Any
    UsageContext = Any

logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint: TypeAlias = type[Any] | object
TypeHintT: TypeAlias = type[T] | object

def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}."
            ) from e

    return _parse_type

def optional_type(return_type: Callable[[str], T]) -> Callable[[str], T | None]:
    def _optional_type(val: str) -> T | None:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type

def union_dict_and_str(val: str) -> str | dict[str, str] | None:
    if not re.match(r"(?s)^\s*{.*}\s*$", val):
        return str(val)
    return optional_type(json.loads)(val)

def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    return any(is_type(type_hint, type) for type_hint in type_hints)

def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:

    If `type_hints` also contains `str`, we use `metavar` instead of `choices`.
    return type_hint.__module__ != "builtins"

def get_type_hints(type_hint: TypeHint) -> set[TypeHint]:

    If `--help` or `mkdocs` are not present in the command line command, the
    attribute documentation will not be included in the help output.

    The heavy computation is cached via functools.lru_cache, and a deep copy
    is returned so callers can mutate the dictionary without affecting the
    cached version.

    model: str = ModelConfig.model
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts
    model_weights: str = ModelConfig.model_weights
    served_model_name: str | list[str] | None = ModelConfig.served_model_name
    tokenizer: str | None = ModelConfig.tokenizer
    hf_config_path: str | None = ModelConfig.hf_config_path
    runner: RunnerOption = ModelConfig.runner
    convert: ConvertOption = ModelConfig.convert
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    tokenizer_mode: TokenizerMode | str = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    allowed_media_domains: list[str] | None = ModelConfig.allowed_media_domains
    download_dir: str | None = LoadConfig.download_dir
    safetensors_load_strategy: str = LoadConfig.safetensors_load_strategy
    load_format: str | LoadFormats = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: ModelDType = ModelConfig.dtype
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: int = ModelConfig.seed
    max_model_len: int | None = ModelConfig.max_model_len
    cudagraph_capture_sizes: list[int] | None = (
        CompilationConfig.cudagraph_capture_sizes
    )
    max_cudagraph_capture_size: int | None = get_field(
        CompilationConfig, "max_cudagraph_capture_size"
    )
    # Parallelism (Fixed to 1 for LiteEngine)
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Compatibility stubs
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    nnodes: int = 1
    node_rank: int = 0
    
    distributed_executor_backend: str = "uni"
    worker_cls: str = "auto"
    
    # MoE compatibility
    enable_expert_parallel: bool = False
    all2all_backend: str = "naive"
    enable_eplb: bool = False
    eplb_config: Any = Field(default_factory=dict)
    expert_placement_strategy: str = "linear"

    # Context Parallel stubs
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    dcp_kv_cache_interleave_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    
    # API stubs
    _api_process_count: int = 1
    _api_process_rank: int = 0
    
    max_parallel_loading_workers: int | None = None
    block_size: BlockSize | None = CacheConfig.block_size
    enable_prefix_caching: bool | None = None
    prefix_caching_hash_algo: PrefixCachingHashAlgo = (
        CacheConfig.prefix_caching_hash_algo
    )
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    swap_space: float = CacheConfig.swap_space
    cpu_offload_gb: float = CacheConfig.cpu_offload_gb
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    kv_cache_memory_bytes: int | None = CacheConfig.kv_cache_memory_bytes
    max_num_batched_tokens: int | None = None
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: int | None = None
    max_logprobs: int = ModelConfig.max_logprobs
    logprobs_mode: LogprobsMode = ModelConfig.logprobs_mode
    disable_log_stats: bool = False
    aggregate_engine_logging: bool = False
    revision: str | None = ModelConfig.revision
    code_revision: str | None = ModelConfig.code_revision
    hf_token: bool | str | None = ModelConfig.hf_token
    hf_overrides: HfOverrides = get_field(ModelConfig, "hf_overrides")
    tokenizer_revision: str | None = ModelConfig.tokenizer_revision
    quantization: QuantizationMethods | None = ModelConfig.quantization
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    limit_mm_per_prompt: dict[str, int | dict[str, int]] = get_field(
        MultiModalConfig, "limit_per_prompt"
    )
    enable_mm_embeds: bool = MultiModalConfig.enable_mm_embeds
    interleave_mm_strings: bool = MultiModalConfig.interleave_mm_strings
    media_io_kwargs: dict[str, dict[str, Any]] = get_field(
        MultiModalConfig, "media_io_kwargs"
    )
    mm_processor_kwargs: dict[str, Any] | None = MultiModalConfig.mm_processor_kwargs
    mm_processor_cache_gb: float = MultiModalConfig.mm_processor_cache_gb
    mm_processor_cache_type: MMCacheType | None = (
        MultiModalConfig.mm_processor_cache_type
    )
    mm_shm_cache_max_object_size_mb: int = (
        MultiModalConfig.mm_shm_cache_max_object_size_mb
    )
    mm_encoder_only: bool = MultiModalConfig.mm_encoder_only
    mm_encoder_tp_mode: MMEncoderTPMode = MultiModalConfig.mm_encoder_tp_mode
    mm_encoder_attn_backend: AttentionBackendEnum | str | None = (
        MultiModalConfig.mm_encoder_attn_backend
    )
    io_processor_plugin: str | None = None
    skip_mm_profiling: bool = MultiModalConfig.skip_mm_profiling
    video_pruning_rate: float = MultiModalConfig.video_pruning_rate
    # LoRA fields
    enable_lora: bool = False
    max_loras: int = LoRAConfig.max_loras
    max_lora_rank: int = LoRAConfig.max_lora_rank
    default_mm_loras: dict[str, str] | None = LoRAConfig.default_mm_loras
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    max_cpu_loras: int | None = LoRAConfig.max_cpu_loras
    lora_dtype: str | torch.dtype | None = LoRAConfig.lora_dtype
    enable_tower_connector_lora: bool = LoRAConfig.enable_tower_connector_lora

    num_gpu_blocks_override: int | None = CacheConfig.num_gpu_blocks_override
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: str | list[str] = get_field(LoadConfig, "ignore_patterns")

    enable_chunked_prefill: bool | None = None
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    disable_hybrid_kv_cache_manager: bool | None = (
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )

    structured_outputs_config: StructuredOutputsConfig = get_field(
        VllmConfig, "structured_outputs_config"
    )
    reasoning_parser: str = StructuredOutputsConfig.reasoning_parser
    reasoning_parser_plugin: str | None = None

    logits_processor_pattern: str | None = ModelConfig.logits_processor_pattern

    show_hidden_metrics_for_version: str | None = (
        ObservabilityConfig.show_hidden_metrics_for_version
    )
    otlp_traces_endpoint: str | None = ObservabilityConfig.otlp_traces_endpoint
    collect_detailed_traces: list[DetailedTraceModules] | None = (
        ObservabilityConfig.collect_detailed_traces
    )
    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics
    kv_cache_metrics_sample: float = get_field(
        ObservabilityConfig, "kv_cache_metrics_sample"
    )
    cudagraph_metrics: bool = ObservabilityConfig.cudagraph_metrics
    enable_layerwise_nvtx_tracing: bool = (
        ObservabilityConfig.enable_layerwise_nvtx_tracing
    )
    enable_mfu_metrics: bool = ObservabilityConfig.enable_mfu_metrics
    enable_logging_iteration_details: bool = (
        ObservabilityConfig.enable_logging_iteration_details
    )
    enable_mm_processor_stats: bool = ObservabilityConfig.enable_mm_processor_stats
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls

    pooler_config: PoolerConfig | None = ModelConfig.pooler_config
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")
    worker_cls: str = "auto"
    worker_extension_cls: str = ""

    generation_config: str = ModelConfig.generation_config
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    override_generation_config: dict[str, Any] = get_field(
        ModelConfig, "override_generation_config"
    )
    model_impl: str = ModelConfig.model_impl
    override_attention_dtype: str = ModelConfig.override_attention_dtype
    attention_backend: AttentionBackendEnum | None = AttentionConfig.backend

    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales
    mamba_cache_dtype: MambaDType = CacheConfig.mamba_cache_dtype
    mamba_ssm_cache_dtype: MambaDType = CacheConfig.mamba_ssm_cache_dtype
    mamba_block_size: int | None = get_field(CacheConfig, "mamba_block_size")
    mamba_cache_mode: MambaCacheMode = CacheConfig.mamba_cache_mode

    kv_offloading_size: float | None = CacheConfig.kv_offloading_size
    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend

    additional_config: dict[str, Any] = get_field(VllmConfig, "additional_config")

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str = LoadConfig.pt_load_map_location

    logits_processors: list[str | type[LogitsProcessor]] | None = (
        ModelConfig.logits_processors
    )

        # Model arguments
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if not ("serve" in sys.argv[1:] and "--help" in sys.argv[1:]):
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--runner", **model_kwargs["runner"])
        model_group.add_argument("--convert", **model_kwargs["convert"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode", **model_kwargs["tokenizer_mode"])
        model_group.add_argument(
            "--trust-remote-code", **model_kwargs["trust_remote_code"]
        )
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path", **model_kwargs["hf_config_path"])
        model_group.add_argument(
            "--allowed-local-media-path", **model_kwargs["allowed_local_media_path"]
        )
        model_group.add_argument(
            "--allowed-media-domains", **model_kwargs["allowed_media_domains"]
        )
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision", **model_kwargs["code_revision"])
        model_group.add_argument(
            "--tokenizer-revision", **model_kwargs["tokenizer_revision"]
        )
        model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q", **model_kwargs["quantization"])
        model_group.add_argument(
            "--allow-deprecated-quantization",
            **model_kwargs["allow_deprecated_quantization"],
        )
        model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
        model_group.add_argument(
            "--enable-return-routed-experts",
            **model_kwargs["enable_return_routed_experts"],
        )
        model_group.add_argument("--max-logprobs", **model_kwargs["max_logprobs"])
        model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])
        model_group.add_argument(
            "--disable-sliding-window", **model_kwargs["disable_sliding_window"]
        )
        model_group.add_argument(
            "--disable-cascade-attn", **model_kwargs["disable_cascade_attn"]
        )
        model_group.add_argument(
            "--skip-tokenizer-init", **model_kwargs["skip_tokenizer_init"]
        )
        model_group.add_argument(
            "--enable-prompt-embeds", **model_kwargs["enable_prompt_embeds"]
        )
        model_group.add_argument(
            "--served-model-name", **model_kwargs["served_model_name"]
        )
        model_group.add_argument("--config-format", **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument(
            "--hf-token",
            type=str,
            nargs="?",
            const=True,
            default=model_kwargs["hf_token"]["default"],
            help=model_kwargs["hf_token"]["help"],
        )
        model_group.add_argument("--hf-overrides", **model_kwargs["hf_overrides"])
        model_group.add_argument("--pooler-config", **model_kwargs["pooler_config"])
        model_group.add_argument(
            "--logits-processor-pattern", **model_kwargs["logits_processor_pattern"]
        )
        model_group.add_argument(
            "--generation-config", **model_kwargs["generation_config"]
        )
        model_group.add_argument(
            "--override-generation-config", **model_kwargs["override_generation_config"]
        )
        model_group.add_argument(
            "--enable-sleep-mode", **model_kwargs["enable_sleep_mode"]
        )
        model_group.add_argument("--model-impl", **model_kwargs["model_impl"])
        model_group.add_argument(
            "--override-attention-dtype", **model_kwargs["override_attention_dtype"]
        )
        model_group.add_argument(
            "--logits-processors", **model_kwargs["logits_processors"]
        )
        model_group.add_argument(
            "--io-processor-plugin", **model_kwargs["io_processor_plugin"]
        )

        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format", **load_kwargs["load_format"])
        load_group.add_argument("--download-dir", **load_kwargs["download_dir"])
        load_group.add_argument(
            "--safetensors-load-strategy", **load_kwargs["safetensors_load_strategy"]
        )
        load_group.add_argument(
            "--model-loader-extra-config", **load_kwargs["model_loader_extra_config"]
        )
        load_group.add_argument("--ignore-patterns", **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load", **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--pt-load-map-location", **load_kwargs["pt_load_map_location"]
        )

        # Attention arguments
        attention_kwargs = get_kwargs(AttentionConfig)
        attention_group = parser.add_argument_group(
            title="AttentionConfig",
            description=AttentionConfig.__doc__,
        )
        attention_group.add_argument(
            "--attention-backend", **attention_kwargs["backend"]
        )

        # Structured outputs arguments
        structured_outputs_kwargs = get_kwargs(StructuredOutputsConfig)
        structured_outputs_group = parser.add_argument_group(
            title="StructuredOutputsConfig",
            description=StructuredOutputsConfig.__doc__,
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser",
            # Choices need to be validated after parsing to include plugins
            **structured_outputs_kwargs["reasoning_parser"],
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser-plugin",
            **structured_outputs_kwargs["reasoning_parser_plugin"],
        )

        # Parallel arguments (Simplified for Single GPU)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description="Configuration for single-GPU execution.",
        )
        parallel_group.add_argument(
            "--pipeline-parallel-size", "-pp", type=int, default=1, help="(Fixed to 1)"
        )
        parallel_group.add_argument(
            "--tensor-parallel-size", "-tp", type=int, default=1, help="(Fixed to 1)"
        )
        parallel_group.add_argument(
            "--data-parallel-size", "-dp", type=int, default=1, help="(Fixed to 1)"
        )
        parallel_group.add_argument(
            "--worker-cls", type=str, default="auto", help="Worker class."
        )
        parallel_group.add_argument(
            "--expert-placement-strategy", type=str, default="linear", help="Expert placement strategy."
        )

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument(
            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]
        )
        cache_group.add_argument(
            "--kv-cache-memory-bytes", **cache_kwargs["kv_cache_memory_bytes"]
        )
        cache_group.add_argument("--swap-space", **cache_kwargs["swap_space"])
        cache_group.add_argument("--kv-cache-dtype", **cache_kwargs["cache_dtype"])
        cache_group.add_argument(
            "--num-gpu-blocks-override", **cache_kwargs["num_gpu_blocks_override"]
        )
        cache_group.add_argument(
            "--enable-prefix-caching",
            **{
                **cache_kwargs["enable_prefix_caching"],
                "default": None,
            },
        )
        cache_group.add_argument(
            "--prefix-caching-hash-algo", **cache_kwargs["prefix_caching_hash_algo"]
        )
        cache_group.add_argument("--cpu-offload-gb", **cache_kwargs["cpu_offload_gb"])
        cache_group.add_argument(
            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]
        )
        cache_group.add_argument(
            "--kv-sharing-fast-prefill", **cache_kwargs["kv_sharing_fast_prefill"]
        )
        cache_group.add_argument(
            "--mamba-cache-dtype", **cache_kwargs["mamba_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-ssm-cache-dtype", **cache_kwargs["mamba_ssm_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-block-size", **cache_kwargs["mamba_block_size"]
        )
        cache_group.add_argument(
            "--mamba-cache-mode", **cache_kwargs["mamba_cache_mode"]
        )
        cache_group.add_argument(
            "--kv-offloading-size", **cache_kwargs["kv_offloading_size"]
        )
        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument(
            "--limit-mm-per-prompt", **multimodal_kwargs["limit_per_prompt"]
        )
        multimodal_group.add_argument(
            "--enable-mm-embeds", **multimodal_kwargs["enable_mm_embeds"]
        )
        multimodal_group.add_argument(
            "--media-io-kwargs", **multimodal_kwargs["media_io_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-kwargs", **multimodal_kwargs["mm_processor_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-gb", **multimodal_kwargs["mm_processor_cache_gb"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-type", **multimodal_kwargs["mm_processor_cache_type"]
        )
        multimodal_group.add_argument(
            "--mm-shm-cache-max-object-size-mb",
            **multimodal_kwargs["mm_shm_cache_max_object_size_mb"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-only", **multimodal_kwargs["mm_encoder_only"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-tp-mode", **multimodal_kwargs["mm_encoder_tp_mode"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-backend",
            **multimodal_kwargs["mm_encoder_attn_backend"],
        )
        multimodal_group.add_argument(
            "--interleave-mm-strings", **multimodal_kwargs["interleave_mm_strings"]
        )
        multimodal_group.add_argument(
            "--skip-mm-profiling", **multimodal_kwargs["skip_mm_profiling"]
        )

        multimodal_group.add_argument(
            "--video-pruning-rate", **multimodal_kwargs["video_pruning_rate"]
        )

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.",
        )
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank", **lora_kwargs["max_lora_rank"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument(
            "--enable-tower-connector-lora",
            **lora_kwargs["enable_tower_connector_lora"],
        )
        lora_group.add_argument("--max-cpu-loras", **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument(
            "--fully-sharded-loras", **lora_kwargs["fully_sharded_loras"]
        )
        lora_group.add_argument("--default-mm-loras", **lora_kwargs["default_mm_loras"])

        # Observability arguments
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"],
        )
        observability_group.add_argument(
            "--otlp-traces-endpoint", **observability_kwargs["otlp_traces_endpoint"]
        )
        # TODO: generalise this special case
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p) for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"],
        )
        observability_group.add_argument(
            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
        )
        observability_group.add_argument(
            "--kv-cache-metrics-sample",
            **observability_kwargs["kv_cache_metrics_sample"],
        )
        observability_group.add_argument(
            "--cudagraph-metrics",
            **observability_kwargs["cudagraph_metrics"],
        )
        observability_group.add_argument(
            "--enable-layerwise-nvtx-tracing",
            **observability_kwargs["enable_layerwise_nvtx_tracing"],
        )
        observability_group.add_argument(
            "--enable-mfu-metrics",
            **observability_kwargs["enable_mfu_metrics"],
        )
        observability_group.add_argument(
            "--enable-logging-iteration-details",
            **observability_kwargs["enable_logging_iteration_details"],
        )

        # Scheduler arguments
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **{
                **scheduler_kwargs["max_num_batched_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-seqs",
            **{
                **scheduler_kwargs["max_num_seqs"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-partial-prefills", **scheduler_kwargs["max_num_partial_prefills"]
        )
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"],
        )
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"],
        )
        # multi-step scheduling has been removed; corresponding arguments
        # are no longer supported.
        scheduler_group.add_argument(
            "--scheduling-policy", **scheduler_kwargs["policy"]
        )
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **{
                **scheduler_kwargs["enable_chunked_prefill"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--disable-chunked-mm-input", **scheduler_kwargs["disable_chunked_mm_input"]
        )
        scheduler_group.add_argument(
            "--scheduler-cls", **scheduler_kwargs["scheduler_cls"]
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
        scheduler_group.add_argument(
            "--async-scheduling", **scheduler_kwargs["async_scheduling"]
        )
        scheduler_group.add_argument(
            "--stream-interval", **scheduler_kwargs["stream_interval"]
        )

        # Compilation arguments
        compilation_kwargs = get_kwargs(CompilationConfig)
        compilation_group = parser.add_argument_group(
            title="CompilationConfig",
            description=CompilationConfig.__doc__,
        )
        compilation_group.add_argument(
            "--cudagraph-capture-sizes", **compilation_kwargs["cudagraph_capture_sizes"]
        )
        compilation_group.add_argument(
            "--max-cudagraph-capture-size",
            **compilation_kwargs["max_cudagraph_capture_size"],
        )

        # vLLM arguments
        vllm_kwargs = get_kwargs(VllmConfig)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument(
            "--compilation-config", "-cc", **vllm_kwargs["compilation_config"]
        )
        vllm_group.add_argument(
            "--attention-config", "-ac", **vllm_kwargs["attention_config"]
        )
        vllm_group.add_argument(
            "--additional-config", **vllm_kwargs["additional_config"]
        )
        vllm_group.add_argument(
            "--structured-outputs-config", **vllm_kwargs["structured_outputs_config"]
        )
        vllm_group.add_argument(
            "--optimization-level", **vllm_kwargs["optimization_level"]
        )

        # Other arguments
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="Disable logging statistics.",
        )

        parser.add_argument(
            "--aggregate-engine-logging",
            action="store_true",
            help="Log aggregate rather than per-engine statistics "
            "when using data parallelism.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

    def create_model_config(self) -> ModelConfig:
        # gguf file needs a specific model loader
        if is_gguf(self.model):
            self.quantization = self.load_format = "gguf"

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        return ModelConfig(
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            enable_mm_embeds=self.enable_mm_embeds,
            interleave_mm_strings=self.interleave_mm_strings,
            media_io_kwargs=self.media_io_kwargs,
            skip_mm_profiling=self.skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            mm_processor_cache_type=self.mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,
            mm_encoder_only=self.mm_encoder_only,
            mm_encoder_tp_mode=self.mm_encoder_tp_mode,
            mm_encoder_attn_backend=self.mm_encoder_attn_backend,
            pooler_config=self.pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            override_attention_dtype=self.override_attention_dtype,
            logits_processors=self.logits_processors,
            video_pruning_rate=self.video_pruning_rate,
            io_processor_plugin=self.io_processor_plugin,
        )

    def validate_tensorizer_args(self):
        from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

        for key in self.model_loader_extra_config:
            if key in TensorizerConfig._fields:
                self.model_loader_extra_config["tensorizer_config"][key] = (
                    self.model_loader_extra_config[key]
                )

    def create_load_config(self) -> LoadConfig:
        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

        if self.load_format == "tensorizer":
            if hasattr(self.model_loader_extra_config, "to_serializable"):
                self.model_loader_extra_config = (
                    self.model_loader_extra_config.to_serializable()
                )
            self.model_loader_extra_config["tensorizer_config"] = {}
            self.model_loader_extra_config["tensorizer_config"]["tensorizer_dir"] = (
                self.model
            )
            self.validate_tensorizer_args()

        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            safetensors_load_strategy=self.safetensors_load_strategy,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:
        current_platform.pre_register_and_update()
        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
        self._set_default_max_num_seqs_and_batched_tokens_args(usage_context, model_config)

        sliding_window = model_config.get_sliding_window() if not is_interleaved(model_config.hf_text_config) else None
        resolved_cache_dtype = resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            swap_space=self.swap_space,
            cache_dtype=resolved_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
            mamba_cache_dtype=self.mamba_cache_dtype,
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
            mamba_block_size=self.mamba_block_size,
            mamba_cache_mode=self.mamba_cache_mode,
            kv_offloading_size=self.kv_offloading_size,
            kv_offloading_backend=self.kv_offloading_backend,
        )

        parallel_config = ParallelConfig(is_moe_model=model_config.is_moe)

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING and self.scheduling_policy == "fcfs":
            self.scheduling_policy = "priority"

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                default_mm_loras=self.default_mm_loras,
                fully_sharded_loras=self.fully_sharded_loras,
                lora_dtype=self.lora_dtype,
                enable_tower_connector_lora=self.enable_tower_connector_lora,
                max_cpu_loras=self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        load_config = self.create_load_config()

        structured_outputs_config = copy.deepcopy(self.structured_outputs_config)
        if self.reasoning_parser: structured_outputs_config.reasoning_parser = self.reasoning_parser
        if self.reasoning_parser_plugin: structured_outputs_config.reasoning_parser_plugin = self.reasoning_parser_plugin

        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
            kv_cache_metrics=self.kv_cache_metrics,
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
            cudagraph_metrics=self.cudagraph_metrics,
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
            enable_mfu_metrics=self.enable_mfu_metrics,
            enable_mm_processor_stats=self.enable_mm_processor_stats,
            enable_logging_iteration_details=self.enable_logging_iteration_details,
        )

        compilation_config = copy.deepcopy(self.compilation_config)
        if self.cudagraph_capture_sizes is not None:
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        if self.max_cudagraph_capture_size is not None:
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )
        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            attention_config=self.attention_config,
            lora_config=lora_config,
            structured_outputs_config=structured_outputs_config,
            observability_config=observability_config,
            compilation_config=compilation_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
        )

        return config

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING and self.scheduling_policy == "fcfs":
            logger.info(
                "Simplification: Forcing scheduler policy to priority for "
                "single-process serving."
            )
            self.scheduling_policy = "priority"

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        if not model_config.is_multimodal_model and self.default_mm_loras:
            raise ValueError(
                "Default modality-specific LoRA(s) were provided for a "
                "non multimodal model"
            )

        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                default_mm_loras=self.default_mm_loras,
                fully_sharded_loras=self.fully_sharded_loras,
                lora_dtype=self.lora_dtype,
                enable_tower_connector_lora=self.enable_tower_connector_lora,
                max_cpu_loras=self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"

        # Attention config overrides
        attention_config = copy.deepcopy(self.attention_config)
        if self.attention_backend is not None:
            if attention_config.backend is not None:
                raise ValueError(
                    "attention_backend and attention_config.backend "
                    "are mutually exclusive"
                )
            # Convert string to enum if needed (CLI parsing returns a string)
            if isinstance(self.attention_backend, str):
                attention_config.backend = AttentionBackendEnum[
                    self.attention_backend.upper()
                ]
            else:
                attention_config.backend = self.attention_backend

        load_config = self.create_load_config()

        # Pass reasoning_parser into StructuredOutputsConfig
        if self.reasoning_parser:
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser

        if self.reasoning_parser_plugin:
            self.structured_outputs_config.reasoning_parser_plugin = (
                self.reasoning_parser_plugin
            )

        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
            kv_cache_metrics=self.kv_cache_metrics,
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
            cudagraph_metrics=self.cudagraph_metrics,
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
            enable_mfu_metrics=self.enable_mfu_metrics,
            enable_mm_processor_stats=self.enable_mm_processor_stats,
            enable_logging_iteration_details=self.enable_logging_iteration_details,
        )

        # Compilation config overrides
        compilation_config = copy.deepcopy(self.compilation_config)
        if self.cudagraph_capture_sizes is not None:
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        if self.max_cudagraph_capture_size is not None:
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )
        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            attention_config=attention_config,
            lora_config=lora_config,
            structured_outputs_config=self.structured_outputs_config,
            observability_config=observability_config,
            compilation_config=compilation_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
        )

        return config

    def _check_feature_supported(self, model_config: ModelConfig):

    enable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        # Initialize plugin to update the parser, for example, The plugin may
        # add a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=AsyncEngineArgs.enable_log_requests,
            help="Enable logging requests.",
        )
        parser.add_argument(
            "--disable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=not AsyncEngineArgs.enable_log_requests,
            help="[DEPRECATED] Disable logging requests.",
            deprecated=True,
        )
        current_platform.pre_register_and_update(parser)
        return parser

def _raise_unsupported_error(feature_name: str):
    msg = (
        f"{feature_name} is not supported. We recommend to "
        f"remove {feature_name} from your config."
    )
    raise NotImplementedError(msg)

def human_readable_int(value: str) -> int:
    value = value.strip()

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)
    if match:
        decimal_multiplier = {
            "k": 10**3,
            "m": 10**6,
            "g": 10**9,
            "t": 10**12,
        }
        binary_multiplier = {
            "K": 2**10,
            "M": 2**20,
            "G": 2**30,
            "T": 2**40,
        }

        number, suffix = match.groups()
        if suffix in decimal_multiplier:
            mult = decimal_multiplier[suffix]
            return int(float(number) * mult)
        elif suffix in binary_multiplier:
            mult = binary_multiplier[suffix]
            # Do not allow decimals with binary multipliers
            try:
                return int(number) * mult
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    "Decimals are not allowed "
                    f"with binary suffixes like {suffix}. Did you mean to use "
                    f"{number}{suffix.lower()} instead?"
                ) from e

    # Regular plain number.
    return int(value)

def human_readable_int_or_auto(value: str) -> int:
    value = value.strip()

    if value == "-1" or value.lower() == "auto":
        return -1

    return human_readable_int(value)
