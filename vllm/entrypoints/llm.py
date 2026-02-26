# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar

from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)
from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    StructuredOutputsConfig,
    is_init_field,
)
from vllm.config.compilation import CompilationMode
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreContentPartParam,
    ScoreMultiModalParam,
    _cosine_similarity,
    _validate_score_input_lens,
    compress_token_type_ids,
    get_score_prompt,
)
from vllm.entrypoints.utils import log_non_default_args
from vllm.inputs import (
    DataPrompt,
    EmbedsPrompt,
    ExplicitEncoderDecoderPrompt,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.inputs.parse import get_prompt_components, is_explicit_encoder_decoder_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils.collection_utils import as_iter, is_list_of
from vllm.utils.counter import Counter
from vllm.engine.v1.llm_engine import LLMEngine
from vllm.sample.logits_processor import LogitsProcessor
from vllm.entrypoints.output_processors.abstract import OutputProcessorStrategy
from vllm.entrypoints.output_processors.default import DefaultOutputProcessor
from vllm.entrypoints.output_processors.verl import VerlOutputProcessor

if TYPE_CHECKING:
    from vllm.metrics.reader import Metric

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)

EnginePrompt: TypeAlias = TextPrompt | TokensPrompt | EmbedsPrompt
EngineEncDecPrompt: TypeAlias = ExplicitEncoderDecoderPrompt[EnginePrompt, EnginePrompt]

class LLM:

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        output_processor_type: str = "default",
        **kwargs: Any,
    ) -> None:
            if value is None:
                return cls()
            if isinstance(value, dict):
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
            return value

        if isinstance(compilation_config, int):
            compilation_config_instance = CompilationConfig(
                mode=CompilationMode(compilation_config)
            )
        else:
            compilation_config_instance = _make_config(
                compilation_config, CompilationConfig
            )

        structured_outputs_instance = _make_config(
            structured_outputs_config, StructuredOutputsConfig
        )
        attention_config_instance = _make_config(attention_config, AttentionConfig)

        # warn about single-process data parallel usage.
        _dp_size = int(kwargs.get("data_parallel_size", 1))
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")
        if (
            _dp_size > 1
            and not _distributed_executor_backend == "external_launcher"
            and not current_platform.is_tpu()
        ):
            raise ValueError(
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/offline_inference/data_parallel.py'."
            )

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            enable_return_routed_experts=enable_return_routed_experts,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            structured_outputs_config=structured_outputs_instance,
            attention_config=attention_config_instance,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )

        log_non_default_args(engine_args)

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)
        self.supported_tasks = supported_tasks

        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor
        self.io_processor = self.llm_engine.io_processor

        # Cache for __repr__ to avoid repeated collective_rpc calls
        self._cached_repr: str | None = None

        # Initialize Output Processor
        if output_processor_type == "verl":
            self.output_processor = VerlOutputProcessor()
        else:
            self.output_processor = DefaultOutputProcessor()

    def get_tokenizer(self) -> TokenizerLike:
        return self.llm_engine.get_tokenizer()

    def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()
        self.llm_engine.reset_mm_cache()

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput] | Any:
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        self._validate_and_add_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=self._get_modality_specific_lora_reqs(prompts, lora_request),
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        validated_outputs = self.engine_class.validate_outputs(outputs, RequestOutput)
        return self.output_processor.process_outputs(validated_outputs)

    def _get_modality_specific_lora_reqs(
        self,
        prompts: PromptType | Sequence[PromptType],
        lora_request: list[LoRARequest] | LoRARequest | None,
    ):
        # Grab the lora config off the vllm config on the engine,
        # since this is the same for both v0 & v1.
        lora_config = self.llm_engine.vllm_config.lora_config

        # If there's no lora config / default_mm_loras, or the model
        # isn't multimodal, leave the lora as is.
        if (
            lora_config is None
            or not self.model_config.is_multimodal_model
            or (lora_config and lora_config.default_mm_loras is None)
        ):
            return lora_request

        if not isinstance(prompts, Sequence) or isinstance(prompts, str):
            prompts = [prompts]

        optional_loras = (
            [lora_request] * len(prompts)
            if not isinstance(lora_request, Sequence)
            else lora_request
        )

        return [
            self._resolve_single_prompt_mm_lora(
                prompt,
                opt_lora_req,
                lora_config.default_mm_loras,
            )
            for prompt, opt_lora_req in zip(prompts, optional_loras)
        ]

    def _resolve_single_prompt_mm_lora(
        self,
        prompt: PromptType,
        lora_request: LoRARequest | None,
        default_mm_loras: dict[str, str] | None,
    ):
        if (
            not default_mm_loras
            or not isinstance(prompt, dict)
            or not (mm_data := prompt.get("multi_modal_data") or {})
        ):
            return lora_request

        intersection = set(
            mm_data.keys()  # type: ignore
        ).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request
        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be"
                " used by a single prompt consuming several modalities; "
                " currently we only support one lora per request; as such,"
                " lora(s) registered with modalities: %s"
                " will be skipped",
                intersection,
            )
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.llm_engine.apply_model(func)

    def _get_beam_search_lora_requests(
        self,
        lora_request: list[LoRARequest] | LoRARequest | None,
        prompts: list[TokensPrompt | TextPrompt],
    ) -> list[LoRARequest | None]:
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `TokensPrompts` objects containing the tokenized prompt
            after chat template interpolation, and the raw multi-modal inputs.
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `TokensPrompts` objects containing the tokenized prompt
            after chat template interpolation, and the raw multi-modal inputs.
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        Generate rewards for each prompt.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache. The content of kv cache
                is forgotten. Level 1 sleep is good for sleeping and waking
                up the engine to run the same model again. The model weights
                are backed up in CPU memory. Please make sure there's enough
                CPU memory to store the model weights. Level 2 sleep will
                discard both the model weights and the kv cache. The content
                of both the model weights and kv cache is forgotten. Level 2
                sleep is good for sleeping and waking up the engine to run a
                different model or update the model, where previous model
                weights are not needed. It reduces CPU memory pressure.
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                engine is used again.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        # Cache the result to avoid repeated collective_rpc calls
        if self._cached_repr is None:
            results = self.llm_engine.collective_rpc("get_model_inspection")
            # In distributed settings, we get results from all workers
            # Just return the first one (they should all be the same)
            if results:
                self._cached_repr = results[0]
            else:
                self._cached_repr = f"LLM(model={self.model_config.model!r})"
        return self._cached_repr
