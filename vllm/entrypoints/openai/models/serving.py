# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from asyncio import Lock
from collections import defaultdict
from http import HTTPStatus

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.utils import sanitize_message
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.utils.counter import AtomicCounter

logger = init_logger(__name__)


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    """

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers():
            self.lora_resolvers.append(
                LoRAResolverRegistry.get_resolver(lora_resolver_name)
            )
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)

        self.input_processor = self.engine_client.input_processor
        self.io_processor = self.engine_client.io_processor
        self.renderer = self.engine_client.renderer
        self.model_config = self.engine_client.model_config
        self.max_model_len = self.model_config.max_model_len

    async def init_static_loras(self):
        """Loads all static LoRA modules."""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            lora_int_id = self.lora_id_counter.inc(1)
            lora_request = LoRARequest(
                lora_name=lora.name,
                lora_int_id=lora_int_id,
                lora_path=lora.path,
            )
            if lora.base_model_name:
                lora_request.base_model_name = lora.base_model_name
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                raise ValueError(str(e)) from e
            self.lora_requests[lora.name] = lora_request

    def is_base_model(self, model_name) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        """Return the base model or a loaded LoRA name."""
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models (base + loaded LoRA)."""
        model_cards = [
            ModelCard(
                id=base_model.name,
                max_model_len=self.max_model_len,
                root=base_model.model_path,
                permission=[ModelPermission()],
            )
            for base_model in self.base_model_paths
        ]
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.lora_path,
                parent=lora.base_model_name
                if lora.base_model_name
                else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    async def resolve_lora(self, lora_name: str) -> LoRARequest | ErrorResponse:
        """Attempt to resolve and load a LoRA adapter by name."""
        async with self.lora_resolver_lock[lora_name]:
            if lora_name in self.lora_requests:
                return self.lora_requests[lora_name]

            base_model_name = self.model_config.model
            unique_id = self.lora_id_counter.inc(1)
            found_adapter = False

            for resolver in self.lora_resolvers:
                lora_request = await resolver.resolve_lora(base_model_name, lora_name)
                if lora_request is None:
                    continue
                found_adapter = True
                lora_request.lora_int_id = unique_id
                try:
                    await self.engine_client.add_lora(lora_request)
                    self.lora_requests[lora_name] = lora_request
                    logger.info(
                        "Resolved and loaded LoRA adapter '%s' using %s",
                        lora_name,
                        resolver.__class__.__name__,
                    )
                    return lora_request
                except Exception as e:
                    logger.warning(
                        "Failed to load LoRA '%s' resolved by %s: %s. "
                        "Trying next resolver.",
                        lora_name,
                        resolver.__class__.__name__,
                        e,
                    )
                    continue

            if found_adapter:
                return create_error_response(
                    message=(
                        f"LoRA adapter '{lora_name}' was found but could not be loaded."
                    ),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            return create_error_response(
                message=f"LoRA adapter {lora_name} does not exist",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

    # LoRA adapter management remains supported in lite mode.


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=err_type,
            code=status_code.value,
        )
    )
