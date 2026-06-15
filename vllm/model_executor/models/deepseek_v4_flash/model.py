# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from .block import (
    DeepSeekV4FlashCompressedLayerReferenceRunner,
    DeepSeekV4FlashSlidingLayerReferenceRunner,
)
from .compressed_kv import DeepSeekV4CompressedKVCache
from .config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
    layer_compress_ratio,
)
from .expert_cache import DeepSeekV4FlashExpertPrefetchRequest
from .gguf_reader import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from .gpu_backend import DeepSeekV4FlashGPUBackend
from .gpu_layers import (
    deepseek_v4_flash_compressed_layer_forward,
    deepseek_v4_flash_sliding_layer_forward,
)
from .gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from .gpu_weight_staging import DeepSeekV4FlashGPUWeightStager
from .profiler import DeepSeekV4FlashProfiler
from .quant import q8_0_matrix_from_gguf_payload, q8_0_matvec
from .weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashWeightStore,
)

_RMS_NORM_EPS = 1e-6
_Q8_0_BLOCK_SIZE = 32
_Q8_0_BLOCK_BYTES = 2 + _Q8_0_BLOCK_SIZE
_OUTPUT_PROJECTION_CHUNK_ROWS = 1024
_READONLY_BUFFER_WARNING = "The given buffer is not writable"


class DeepSeekV4FlashForCausalLM(nn.Module):
    """DeepSeek V4 Flash GGUF-backed bring-up model.

    The current forward is intentionally limited to a real tensor smoke path:
    token embedding, final RMSNorm, and output projection. It does not execute
    transformer layers yet.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE,
        weight_store: DeepSeekV4FlashWeightStore | None = None,
        runtime_budget: DeepSeekV4FlashRuntimeBudget | None = None,
        gpu_backend: DeepSeekV4FlashGPUBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.shape = shape
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget
        self.gpu_backend = gpu_backend or DeepSeekV4FlashGPUBackend()
        self.limited_forward_smoke_only = True
        self.reference_execution_available = True
        self.kernel_execution_available = self.gpu_backend.is_ready
        self._gpu_weight_stager: DeepSeekV4FlashGPUWeightStager | None = None
        self._gpu_weight_stager_store_id: int | None = None
        self._gpu_weight_stager_device: torch.device | None = None
        self._deepseek_profiler = DeepSeekV4FlashProfiler(
            enabled=False,
            sync_fn=self._sync_deepseek_profile_device,
        )

    def attach_weight_store(
        self,
        weight_store: DeepSeekV4FlashWeightStore,
        runtime_budget: DeepSeekV4FlashRuntimeBudget,
    ) -> None:
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget
        self._gpu_weight_stager = None
        self._gpu_weight_stager_store_id = None
        self._gpu_weight_stager_device = None

    def close(self) -> None:
        if self.weight_store is not None:
            self.weight_store.close()
            self.weight_store = None
        self._gpu_weight_stager = None
        self._gpu_weight_stager_store_id = None
        self._gpu_weight_stager_device = None

    def enable_deepseek_profile(self, enabled: bool = True) -> None:
        self._deepseek_profiler.enabled = enabled

    def deepseek_profile(self) -> dict[str, object]:
        return self._deepseek_profiler.to_dict()

    def _sync_deepseek_profile_device(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _prefetch_grouped_experts_best_effort(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        layer: DeepSeekV4FlashGroupedExpertTensors,
        expert_ids: tuple[int, ...],
        *,
        layer_idx: int,
    ) -> None:
        try:
            stager.prefetch_grouped_experts(
                layer,
                DeepSeekV4FlashExpertPrefetchRequest(
                    layer_idx=layer_idx,
                    expert_ids=expert_ids,
                ),
            )
        except Exception:
            self._deepseek_profiler.add_counter("deepseek_prefetch_failures", 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        if self.weight_store is None:
            raise RuntimeError(
                "DeepSeekV4FlashForCausalLM requires an attached GGUF weight store"
            )
        if input_ids.ndim != 1:
            raise ValueError(
                "DeepSeekV4FlashForCausalLM.forward only supports 1-D input_ids "
                f"for batch=1 smoke; got {input_ids.ndim}-D"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"input_ids must use an integer dtype; got {input_ids.dtype}"
            )
        if self.runtime_budget is not None:
            context_length = self.runtime_budget.context.context_length
            if context_length > 8192:
                raise ValueError(
                    "DeepSeekV4FlashForCausalLM first release supports context <= "
                    f"8192; got runtime context {context_length}"
                )
        if input_ids.numel() == 0:
            return torch.empty((0, self.shape.vocab_size), dtype=torch.float32)
        if input_ids.numel() != 1:
            raise ValueError(
                "DeepSeekV4FlashForCausalLM limited smoke supports one token until "
                "full transformer forward is implemented; "
                f"got {input_ids.numel()} input tokens"
            )
        return self._forward_embedding_output_projection_smoke_batch_one(input_ids)

    def _forward_embedding_output_projection_smoke_batch_one(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a limited real-tensor projection smoke, not transformer layers."""
        store = self._require_weight_store()
        logits = [
            self._project_one_token_embedding_smoke(store, int(token_id))
            for token_id in input_ids.detach().cpu().tolist()
        ]
        return torch.stack(logits, dim=0)

    def _require_weight_store(self) -> DeepSeekV4FlashWeightStore:
        if self.weight_store is None:
            raise RuntimeError(
                "DeepSeekV4FlashForCausalLM requires an attached GGUF weight store"
            )
        return self.weight_store

    def forward_full_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        store = self._require_weight_store()
        self._validate_full_reference_input(input_ids)
        token_id = int(input_ids.detach().cpu()[0])
        hidden = self._read_token_embedding(store, token_id)
        streams = hidden.reshape(1, -1).repeat(4, 1)
        cache = DeepSeekV4CompressedKVCache(
            context_length=4096,
            hidden_size=self.shape.head_dim,
        )
        for layer_idx in range(self.shape.num_layers):
            if layer_compress_ratio(layer_idx) == 0:
                runner = DeepSeekV4FlashSlidingLayerReferenceRunner(
                    store,
                    layer_idx=layer_idx,
                    shape=self.shape,
                )
                streams = runner.forward(
                    streams,
                    token_id=token_id,
                    token_idx=0,
                )
            else:
                runner = DeepSeekV4FlashCompressedLayerReferenceRunner(
                    store,
                    layer_idx=layer_idx,
                    shape=self.shape,
                )
                streams = runner.forward(
                    streams,
                    token_id=token_id,
                    token_idx=0,
                    cache=cache,
                )
        embd = self._collapse_output_hc(store, streams)
        normalized = self._rms_norm(embd, self._read_output_norm(store))
        self.limited_forward_smoke_only = False
        return self._q8_0_output_projection(store, normalized).reshape(1, -1)

    def forward_kernel(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._require_weight_store()
        token_id, device = self._validate_forward_kernel_input(input_ids)
        self.gpu_backend.require_ready()
        state = DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=self._kernel_context_length(),
                hidden_size=self.shape.hidden_size,
                batch_size=1,
                kv_width=self.shape.head_dim,
                device=device,
            )
        )
        return self._forward_kernel_token_step(
            token_id=token_id,
            state=state,
            token_idx=state.token_position,
            device=device,
        )

    def _forward_kernel_token_step(
        self,
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        store, stager, hidden = self._forward_kernel_token_hidden(
            token_id=token_id,
            state=state,
            token_idx=token_idx,
            device=device,
        )
        with self._deepseek_profiler.section(
            "output_projection",
            token_idx=token_idx,
        ):
            streams = self._kernel_output_streams(hidden)
            logits = self._output_logits_chunked_cuda(
                store,
                stager=stager,
                streams=streams,
                device=device,
            )
        if not logits.is_cuda:
            raise RuntimeError("DeepSeek V4 Flash GPU forward returned CPU logits")
        state.advance_token()
        return logits

    def _forward_kernel_token_step_token_id(
        self,
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        store, stager, hidden = self._forward_kernel_token_hidden(
            token_id=token_id,
            state=state,
            token_idx=token_idx,
            device=device,
        )
        with self._deepseek_profiler.section(
            "output_projection",
            token_idx=token_idx,
        ):
            streams = self._kernel_output_streams(hidden)
            next_token = self._output_token_argmax_chunked_cuda(
                store,
                stager=stager,
                streams=streams,
                device=device,
            )
        if not next_token.is_cuda:
            raise RuntimeError("DeepSeek V4 Flash GPU forward returned CPU token id")
        state.advance_token()
        return next_token.to(device=device, dtype=torch.long).reshape(())

    def _forward_kernel_token_step_token_tensor(
        self,
        *,
        token_id_tensor: torch.Tensor,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        store, stager, hidden = self._forward_kernel_token_hidden_token_tensor(
            token_id_tensor=token_id_tensor,
            state=state,
            token_idx=token_idx,
            device=device,
        )
        with self._deepseek_profiler.section(
            "output_projection",
            token_idx=token_idx,
        ):
            streams = self._kernel_output_streams(hidden)
            next_token = self._output_token_argmax_chunked_cuda(
                store,
                stager=stager,
                streams=streams,
                device=device,
            )
        if not next_token.is_cuda:
            raise RuntimeError("DeepSeek V4 Flash GPU forward returned CPU token id")
        state.advance_token()
        return next_token.to(device=device, dtype=torch.long).reshape(())

    def _forward_kernel_token_hidden(
        self,
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> tuple[
        DeepSeekV4FlashWeightStore,
        DeepSeekV4FlashGPUWeightStager,
        torch.Tensor,
    ]:
        store = self._require_weight_store()
        self.gpu_backend.require_ready()
        state.require_capacity(token_idx)
        if token_idx != state.token_position:
            raise ValueError(
                "forward kernel token_idx must match request state token_position; "
                f"got token_idx={token_idx}, token_position={state.token_position}"
            )
        if token_id < 0 or token_id >= self.shape.vocab_size:
            raise ValueError(
                f"input token id {token_id} is outside vocab range "
                f"[0, {self.shape.vocab_size})"
            )

        stager = self._get_gpu_weight_stager(device)
        hidden = self._stage_token_embedding_cuda(store, token_id, device=device)

        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires layer bindings")
        for layer in layers:
            if layer.layer_index < 2:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_sliding",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_sliding_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        token_idx=token_idx,
                        token_id=token_id,
                    )
            else:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_compressed",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_compressed_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        state=state,
                        token_idx=token_idx,
                        token_id=token_id,
                    )

        return store, stager, hidden

    def _forward_kernel_token_hidden_token_tensor(
        self,
        *,
        token_id_tensor: torch.Tensor,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> tuple[
        DeepSeekV4FlashWeightStore,
        DeepSeekV4FlashGPUWeightStager,
        torch.Tensor,
    ]:
        store = self._require_weight_store()
        self.gpu_backend.require_ready()
        state.require_capacity(token_idx)
        if token_idx != state.token_position:
            raise ValueError(
                "forward kernel token_idx must match request state token_position; "
                f"got token_idx={token_idx}, token_position={state.token_position}"
            )

        token_id_tensor = self._validate_cuda_scalar_token_id_tensor(
            token_id_tensor,
            device=device,
        )
        stager = self._get_gpu_weight_stager(device)
        hidden = self._stage_token_embedding_tensor_cuda(
            store,
            stager,
            token_id_tensor,
            device=device,
        )

        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires layer bindings")
        for layer in layers:
            if layer.layer_index < 2:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_sliding",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_sliding_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        token_idx=token_idx,
                        token_id_tensor=token_id_tensor,
                    )
            else:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_compressed",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_compressed_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        state=state,
                        token_idx=token_idx,
                        token_id_tensor=token_id_tensor,
                    )

        return store, stager, hidden

    def forward_full(
        self,
        input_ids: torch.Tensor,
        *,
        use_kernel: bool = False,
    ) -> torch.Tensor:
        if use_kernel:
            if not self.kernel_execution_available:
                raise NotImplementedError(
                    "DeepSeek V4 Flash kernel execution is not available"
                )
            self.gpu_backend.require_ready()
            return self.forward_kernel(input_ids)
        return self.forward_full_reference(input_ids)

    def generate_greedy_reference(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int = 1,
    ) -> torch.Tensor:
        if max_tokens != 1:
            raise ValueError(
                "generate_greedy_reference currently supports max_tokens=1; "
                f"got {max_tokens}"
            )
        logits = self.forward_full(input_ids, use_kernel=False)
        next_token = torch.argmax(logits[0], dim=-1).to(torch.long)
        return torch.cat([input_ids.to(torch.long), next_token.reshape(1)])

    def generate_greedy_kernel(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int = 1,
    ) -> torch.Tensor:
        with self._deepseek_profiler.section(
            "generate_greedy_kernel",
            input_tokens=int(input_ids.numel()),
            max_tokens=max_tokens,
        ):
            self._require_weight_store()
            device = self._validate_generate_greedy_kernel_input(
                input_ids,
                max_tokens=max_tokens,
            )
            self.gpu_backend.require_ready()
            state = DeepSeekV4FlashGPURequestState(
                DeepSeekV4FlashGPUCacheConfig(
                    context_length=self._kernel_context_length(),
                    hidden_size=self.shape.hidden_size,
                    batch_size=1,
                    kv_width=self.shape.head_dim,
                    device=device,
                )
            )

            output_ids = input_ids.to(device=device, dtype=torch.long).clone()
            prompt_token_ids = output_ids.detach().cpu().tolist()
            next_token_tensor: torch.Tensor | None = None
            for token_id in prompt_token_ids:
                next_token_tensor = self._forward_kernel_token_step_token_id(
                    token_id=int(token_id),
                    state=state,
                    token_idx=state.token_position,
                    device=device,
                )

            if next_token_tensor is None:
                raise ValueError(
                    "generate_greedy_kernel requires at least one input token"
                )

            for generated_idx in range(max_tokens):
                output_ids = torch.cat([output_ids, next_token_tensor.reshape(1)])
                if generated_idx == max_tokens - 1:
                    break
                next_token_tensor = self._forward_kernel_token_step_token_tensor(
                    token_id_tensor=next_token_tensor,
                    state=state,
                    token_idx=state.token_position,
                    device=device,
                )

            return output_ids

    def _validate_full_reference_input(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 1:
            raise ValueError(
                "forward_full_reference supports a 1-D batch=1 token vector; "
                f"got {input_ids.ndim}-D"
            )
        if input_ids.numel() != 1:
            raise ValueError(
                "forward_full_reference currently supports exactly one token; "
                f"got {input_ids.numel()}"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"input_ids must use an integer dtype; got {input_ids.dtype}"
            )

    def _validate_forward_kernel_input(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[int, torch.device]:
        if input_ids.ndim != 1:
            raise ValueError(
                "forward_kernel supports a 1-D batch=1 token vector; "
                f"got {input_ids.ndim}-D"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"input_ids must use an integer dtype; got {input_ids.dtype}"
            )
        if not input_ids.is_cuda:
            raise ValueError("forward_kernel requires CUDA input_ids")
        context_length = self._kernel_context_length()
        if input_ids.numel() > context_length:
            raise ValueError(
                "forward_kernel input context exceeds configured budget: "
                f"{input_ids.numel()} tokens > {context_length}"
            )
        if input_ids.numel() != 1:
            raise ValueError(
                "forward_kernel currently supports exactly one batch=1 token; "
                f"got {input_ids.numel()} tokens"
            )
        token_id = int(input_ids.item())
        if token_id < 0 or token_id >= self.shape.vocab_size:
            raise ValueError(
                f"input token id {token_id} is outside vocab range "
                f"[0, {self.shape.vocab_size})"
            )
        return token_id, input_ids.device

    def _validate_generate_greedy_kernel_input(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
    ) -> torch.device:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive; got {max_tokens}")
        if input_ids.ndim != 1:
            raise ValueError(
                "generate_greedy_kernel supports a 1-D batch=1 token vector; "
                f"got {input_ids.ndim}-D"
            )
        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                f"input_ids must use an integer dtype; got {input_ids.dtype}"
            )
        if not input_ids.is_cuda:
            raise ValueError("generate_greedy_kernel requires CUDA input_ids")
        if input_ids.numel() == 0:
            raise ValueError("generate_greedy_kernel requires at least one input token")
        context_length = self._kernel_context_length()
        total_tokens = input_ids.numel() + max_tokens
        if total_tokens > context_length:
            raise ValueError(
                "generate_greedy_kernel input plus generated tokens exceeds "
                f"configured budget: {total_tokens} tokens > {context_length}"
            )
        for token_id in input_ids.detach().cpu().tolist():
            token_id_int = int(token_id)
            if token_id_int < 0 or token_id_int >= self.shape.vocab_size:
                raise ValueError(
                    f"input token id {token_id_int} is outside vocab range "
                    f"[0, {self.shape.vocab_size})"
                )
        return input_ids.device

    def _kernel_context_length(self) -> int:
        if self.runtime_budget is None:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward requires a runtime budget"
            )
        context_length = self.runtime_budget.context.context_length
        if context_length <= 0:
            raise ValueError(
                f"runtime context length must be positive; got {context_length}"
            )
        return context_length

    def _get_gpu_weight_stager(
        self,
        device: torch.device,
    ) -> DeepSeekV4FlashGPUWeightStager:
        store = self._require_weight_store()
        device = self._canonical_gpu_staging_device(device)
        store_id = id(store)
        if (
            self._gpu_weight_stager is None
            or self._gpu_weight_stager_store_id != store_id
            or self._gpu_weight_stager_device != device
        ):
            self._gpu_weight_stager = DeepSeekV4FlashGPUWeightStager(
                store,
                device=device,
                max_staged_bytes=self._gpu_staging_budget_bytes(),
            )
            self._gpu_weight_stager_store_id = store_id
            self._gpu_weight_stager_device = device
        return self._gpu_weight_stager

    @staticmethod
    def _canonical_gpu_staging_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type != "cuda":
            return device
        if device.index is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cuda", 0)

    def _gpu_staging_budget_bytes(self) -> int | None:
        if self.runtime_budget is None:
            return None
        return max(
            0,
            self.runtime_budget.available_headroom_bytes
            - self.runtime_budget.min_system_headroom_bytes,
        )

    def gpu_staging_memory_stats(self) -> dict[str, int | None]:
        stager = self._gpu_weight_stager
        if stager is None:
            return {
                "staged_bytes": 0,
                "max_staged_bytes": self._gpu_staging_budget_bytes(),
                "dynamic_entries": 0,
                "grouped_entries": 0,
            }
        return stager.memory_stats()

    def prepare_for_serving(
        self,
        *,
        context_length: int,
        device: torch.device,
    ) -> dict[str, int | None]:
        kernel_context_length = self._kernel_context_length()
        if context_length != kernel_context_length:
            raise ValueError(
                "prepare_for_serving context_length must match runtime budget: "
                f"requested context_length={context_length}, "
                f"runtime context_length={kernel_context_length}"
            )
        store = self._require_weight_store()
        stager = self._get_gpu_weight_stager(device)
        bindings = store.bindings

        output_hyper_connection = getattr(
            bindings,
            "output_hyper_connection",
            None,
        )
        if output_hyper_connection is not None:
            self._stage_output_hyper_connection_cuda(
                stager,
                stream_elements=4 * self.shape.hidden_size,
            )

        output_norm = getattr(bindings, "output_norm", None)
        if output_norm is not None:
            stager.stage_vector(output_norm)

        return stager.memory_stats()

    def _stage_token_embedding_cuda(
        self,
        store: DeepSeekV4FlashWeightStore,
        token_id: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        hidden = self._read_token_embedding(store, token_id)
        return hidden.to(device=device, dtype=torch.float32, non_blocking=True)

    def _stage_token_embedding_tensor_cuda(
        self,
        store: DeepSeekV4FlashWeightStore,
        stager: DeepSeekV4FlashGPUWeightStager,
        token_id_tensor: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = store.bindings.token_embedding
        if tensor.tensor_type != GGML_TYPE_F16:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects F16 token_embd.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        hidden_size = self.shape.hidden_size
        vocab_size = self.shape.vocab_size
        if tensor.dims != (hidden_size, vocab_size):
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects token_embd.weight dims "
                f"({hidden_size}, {vocab_size}); got {tensor.dims}"
            )

        token_id_tensor = self._validate_cuda_scalar_token_id_tensor(
            token_id_tensor,
            device=device,
        )
        embedding_matrix = stager.stage_matrix(tensor).reshape(vocab_size, hidden_size)
        hidden = torch.index_select(
            embedding_matrix,
            0,
            token_id_tensor.reshape(1),
        ).reshape(hidden_size)
        return hidden.to(dtype=torch.float32)

    def _validate_cuda_scalar_token_id_tensor(
        self,
        token_id_tensor: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if token_id_tensor.ndim != 0:
            raise ValueError(
                "generated token id tensor must be scalar; "
                f"got {token_id_tensor.ndim}-D"
            )
        if not token_id_tensor.is_cuda:
            raise ValueError("generated token id tensor must be CUDA")
        if token_id_tensor.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                "generated token id tensor must use an integer dtype; "
                f"got {token_id_tensor.dtype}"
            )

        token_id_tensor = token_id_tensor.to(device=device, dtype=torch.long)
        token_id_tensor = token_id_tensor.reshape(())
        # Generated ids normally come from CUDA argmax. Keep range validation on
        # device so the decode carry path does not synchronize back to CPU.
        torch._assert_async(
            (token_id_tensor >= 0) & (token_id_tensor < self.shape.vocab_size),
            f"generated token id is outside vocab range [0, {self.shape.vocab_size})",
        )
        return token_id_tensor

    def _kernel_output_streams(self, hidden: torch.Tensor) -> torch.Tensor:
        if not hidden.is_cuda:
            raise RuntimeError("DeepSeek V4 Flash GPU layer returned CPU hidden state")
        if hidden.ndim == 1:
            if hidden.shape != (self.shape.hidden_size,):
                raise ValueError(
                    "forward_kernel hidden shape must match hidden_size; "
                    f"got {tuple(hidden.shape)}"
                )
            return hidden.to(torch.float32).reshape(1, -1).expand(4, -1).clone()
        if hidden.ndim != 2:
            raise ValueError(
                "forward_kernel hidden must be 1-D or mHC stream-shaped 2-D; "
                f"got {hidden.ndim}-D"
            )
        if hidden.shape != (4, self.shape.hidden_size):
            raise ValueError(
                "forward_kernel mHC streams must have shape "
                f"(4, {self.shape.hidden_size}); got {tuple(hidden.shape)}"
            )
        return hidden.to(torch.float32)

    def _stage_output_hyper_connection_cuda(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        *,
        stream_elements: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_hc = self._require_weight_store().bindings.output_hyper_connection
        if output_hc is None:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward requires output hyper-connection"
            )
        weight = stager.stage_matrix(output_hc.fn)
        expected = (4, stream_elements)
        if tuple(weight.shape) == (stream_elements, 4):
            weight = weight.T.contiguous()
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"output_hc_fn staged shape must be {expected}; "
                f"got {tuple(weight.shape)}"
            )
        return (
            weight,
            stager.stage_vector(output_hc.scale),
            stager.stage_vector(output_hc.base),
        )

    def _required_output_norm_tensor(
        self,
        store: DeepSeekV4FlashWeightStore,
    ) -> DeepSeekV4FlashTensor:
        tensor = store.bindings.output_norm
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires output_norm")
        return tensor

    def _output_logits_chunked_cuda(
        self,
        store: DeepSeekV4FlashWeightStore,
        *,
        stager: DeepSeekV4FlashGPUWeightStager,
        streams: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = store.bindings.output_head
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires output.weight")
        if tensor.tensor_type != GGML_TYPE_Q8_0:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects Q8_0 output.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        columns = self.shape.hidden_size
        rows = self.shape.vocab_size
        if tensor.dims != (columns, rows):
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects output.weight dims "
                f"({columns}, {rows}); got {tensor.dims}"
            )
        if columns % _Q8_0_BLOCK_SIZE != 0:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward requires hidden size divisible by "
                f"{_Q8_0_BLOCK_SIZE}; got {columns}"
            )

        output_hc_weight, output_hc_scale, output_hc_base = (
            self._stage_output_hyper_connection_cuda(
                stager,
                stream_elements=streams.numel(),
            )
        )
        output_norm_weight = stager.stage_vector(
            self._required_output_norm_tensor(store),
        )
        blocks_per_row = columns // _Q8_0_BLOCK_SIZE
        row_bytes = blocks_per_row * _Q8_0_BLOCK_BYTES
        logits_chunks: list[torch.Tensor] = []
        payload: memoryview | None = None
        try:
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                cached = stager.get_output_q8_chunk(
                    tensor,
                    row_start=row_start,
                    row_end=row_end,
                )
                if cached is None:
                    if payload is None:
                        payload = store.tensor_payload(tensor)
                    chunk_rows = row_end - row_start
                    byte_start = row_start * row_bytes
                    byte_end = row_end * row_bytes
                    chunk_view = payload[byte_start:byte_end]
                    try:
                        values, scales = q8_0_matrix_from_gguf_payload(
                            chunk_view,
                            rows=chunk_rows,
                            columns=columns,
                            block_size=_Q8_0_BLOCK_SIZE,
                        )
                    finally:
                        chunk_view.release()
                    lm_head_values, lm_head_scales = stager.stage_output_q8_chunk(
                        tensor,
                        row_start=row_start,
                        row_end=row_end,
                        values=values,
                        scales=scales,
                    )
                else:
                    lm_head_values, lm_head_scales = cached
                chunk_logits = self.gpu_backend.output_logits(
                    streams=streams,
                    lm_head_values=lm_head_values,
                    lm_head_scales=lm_head_scales,
                    output_hc_weight=output_hc_weight,
                    output_hc_scale=output_hc_scale,
                    output_hc_base=output_hc_base,
                    output_norm_weight=output_norm_weight,
                    block_size=_Q8_0_BLOCK_SIZE,
                )
                if not chunk_logits.is_cuda:
                    raise RuntimeError(
                        "DeepSeek V4 Flash GPU output chunk returned CPU logits"
                    )
                logits_chunks.append(chunk_logits.reshape(-1))
        finally:
            if payload is not None:
                payload.release()
        if not logits_chunks:
            raise RuntimeError("DeepSeek V4 Flash GPU output produced no chunks")
        return torch.cat(logits_chunks, dim=0).reshape(1, -1)

    def _output_token_argmax_chunked_cuda(
        self,
        store: DeepSeekV4FlashWeightStore,
        *,
        stager: DeepSeekV4FlashGPUWeightStager,
        streams: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = store.bindings.output_head
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires output.weight")
        if tensor.tensor_type != GGML_TYPE_Q8_0:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects Q8_0 output.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        columns = self.shape.hidden_size
        rows = self.shape.vocab_size
        if tensor.dims != (columns, rows):
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward expects output.weight dims "
                f"({columns}, {rows}); got {tensor.dims}"
            )
        if columns % _Q8_0_BLOCK_SIZE != 0:
            raise RuntimeError(
                "DeepSeek V4 Flash GPU forward requires hidden size divisible by "
                f"{_Q8_0_BLOCK_SIZE}; got {columns}"
            )

        output_hc_weight, output_hc_scale, output_hc_base = (
            self._stage_output_hyper_connection_cuda(
                stager,
                stream_elements=streams.numel(),
            )
        )
        output_norm_weight = stager.stage_vector(
            self._required_output_norm_tensor(store),
        )
        blocks_per_row = columns // _Q8_0_BLOCK_SIZE
        row_bytes = blocks_per_row * _Q8_0_BLOCK_BYTES
        payload: memoryview | None = None
        best_value: torch.Tensor | None = None
        best_token: torch.Tensor | None = None
        try:
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                cached = stager.get_output_q8_chunk(
                    tensor,
                    row_start=row_start,
                    row_end=row_end,
                )
                if cached is None:
                    if payload is None:
                        payload = store.tensor_payload(tensor)
                    chunk_rows = row_end - row_start
                    byte_start = row_start * row_bytes
                    byte_end = row_end * row_bytes
                    chunk_view = payload[byte_start:byte_end]
                    try:
                        values, scales = q8_0_matrix_from_gguf_payload(
                            chunk_view,
                            rows=chunk_rows,
                            columns=columns,
                            block_size=_Q8_0_BLOCK_SIZE,
                        )
                    finally:
                        chunk_view.release()
                    lm_head_values, lm_head_scales = stager.stage_output_q8_chunk(
                        tensor,
                        row_start=row_start,
                        row_end=row_end,
                        values=values,
                        scales=scales,
                    )
                else:
                    lm_head_values, lm_head_scales = cached

                chunk_token: torch.Tensor | None = None
                if hasattr(self.gpu_backend, "output_argmax"):
                    token = self.gpu_backend.output_argmax(
                        streams=streams,
                        lm_head_values=lm_head_values,
                        lm_head_scales=lm_head_scales,
                        output_hc_weight=output_hc_weight,
                        output_hc_scale=output_hc_scale,
                        output_hc_base=output_hc_base,
                        output_norm_weight=output_norm_weight,
                        block_size=_Q8_0_BLOCK_SIZE,
                        row_offset=row_start,
                    )
                    if not token.is_cuda:
                        raise RuntimeError(
                            "DeepSeek V4 Flash GPU output chunk returned CPU token id"
                        )
                    chunk_token = token.to(device=device, dtype=torch.long).reshape(())

                chunk_logits = self.gpu_backend.output_logits(
                    streams=streams,
                    lm_head_values=lm_head_values,
                    lm_head_scales=lm_head_scales,
                    output_hc_weight=output_hc_weight,
                    output_hc_scale=output_hc_scale,
                    output_hc_base=output_hc_base,
                    output_norm_weight=output_norm_weight,
                    block_size=_Q8_0_BLOCK_SIZE,
                ).reshape(-1)
                if not chunk_logits.is_cuda:
                    raise RuntimeError(
                        "DeepSeek V4 Flash GPU output chunk returned CPU logits"
                    )
                if chunk_token is None:
                    chunk_value, chunk_index = torch.max(chunk_logits, dim=0)
                    chunk_token = chunk_index.to(torch.long) + row_start
                else:
                    chunk_index = chunk_token - row_start
                    chunk_value = chunk_logits[chunk_index]
                if best_value is None or best_token is None:
                    best_value = chunk_value
                    best_token = chunk_token
                else:
                    take_chunk = chunk_value > best_value
                    best_value = torch.where(take_chunk, chunk_value, best_value)
                    best_token = torch.where(take_chunk, chunk_token, best_token)
        finally:
            if payload is not None:
                payload.release()
        if best_token is None:
            raise RuntimeError("DeepSeek V4 Flash GPU output produced no chunks")
        return best_token.to(device=device, dtype=torch.long).reshape(())

    def _collapse_output_hc(
        self,
        store: DeepSeekV4FlashWeightStore,
        streams: torch.Tensor,
    ) -> torch.Tensor:
        if streams.shape != (4, self.shape.hidden_size):
            raise ValueError(
                f"output HC streams shape must be (4, {self.shape.hidden_size}); "
                f"got {tuple(streams.shape)}"
            )
        hc = store.bindings.output_hyper_connection
        if hc is None:
            raise RuntimeError("DeepSeek V4 Flash full reference requires output HC")
        flat = streams.reshape(-1).to(torch.float32)
        flat = flat * torch.rsqrt(flat.pow(2).mean() + _RMS_NORM_EPS)
        pre = store.decode_matrix(hc.fn).transpose(0, 1).to(torch.float32).matmul(
            flat
        )
        scale = store.tensor_to_torch(hc.scale, dtype=torch.float32)
        base = store.tensor_to_torch(hc.base, dtype=torch.float32)
        if scale.shape != (1,) or base.shape != (4,):
            raise RuntimeError(
                "DeepSeek V4 Flash output HC expects scale=(1,) and base=(4,)"
            )
        weights = torch.sigmoid(pre * scale[0] + base) + 1e-6
        return (weights.reshape(4, 1) * streams.to(torch.float32)).sum(dim=0)

    def _project_one_token_embedding_smoke(
        self,
        store: DeepSeekV4FlashWeightStore,
        token_id: int,
    ) -> torch.Tensor:
        hidden = self._read_token_embedding(store, token_id)
        norm_weight = self._read_output_norm(store)
        normalized = self._rms_norm(hidden, norm_weight)
        return self._q8_0_output_projection(store, normalized)

    def _read_token_embedding(
        self,
        store: DeepSeekV4FlashWeightStore,
        token_id: int,
    ) -> torch.Tensor:
        tensor = store.bindings.token_embedding
        if tensor.tensor_type != GGML_TYPE_F16:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects F16 token_embd.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        hidden_size = self.shape.hidden_size
        vocab_size = self.shape.vocab_size
        if tensor.dims != (hidden_size, vocab_size):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects token_embd.weight dims "
                f"({hidden_size}, {vocab_size}); got {tensor.dims}"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"input token id {token_id} is outside vocab range [0, {vocab_size})"
            )

        row_bytes = hidden_size * 2
        row_start = token_id * row_bytes
        payload = store.tensor_payload(tensor)
        row_view = payload[row_start : row_start + row_bytes]
        try:
            return _torch_from_readonly_buffer(row_view, dtype=torch.float16).to(
                torch.float32
            )
        finally:
            row_view.release()
            payload.release()

    def _read_output_norm(self, store: DeepSeekV4FlashWeightStore) -> torch.Tensor:
        tensor = store.bindings.output_norm
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash smoke requires output_norm.weight")
        if tensor.tensor_type != GGML_TYPE_F32:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects F32 output_norm.weight; "
                f"got GGML type {tensor.tensor_type}"
            )
        if tensor.dims != (self.shape.hidden_size,):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects output_norm.weight dims "
                f"({self.shape.hidden_size},); got {tensor.dims}"
            )
        payload = store.tensor_payload(tensor)
        try:
            return _torch_from_readonly_buffer(payload, dtype=torch.float32).clone()
        finally:
            payload.release()

    def _rms_norm(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = hidden.to(torch.float32).pow(2).mean()
        scale = torch.rsqrt(variance + _RMS_NORM_EPS)
        return hidden.to(torch.float32) * scale * weight.to(torch.float32)

    def _q8_0_output_projection(
        self,
        store: DeepSeekV4FlashWeightStore,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        tensor = store.bindings.output_head
        if tensor is None:
            raise RuntimeError("DeepSeek V4 Flash smoke requires output.weight")
        if tensor.tensor_type != GGML_TYPE_Q8_0:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects Q8_0 output.weight; "
                f"got GGML type {tensor.tensor_type}"
            )

        columns = self.shape.hidden_size
        rows = self.shape.vocab_size
        if tensor.dims != (columns, rows):
            raise RuntimeError(
                "DeepSeek V4 Flash smoke expects output.weight dims "
                f"({columns}, {rows}); got {tensor.dims}"
            )
        if columns % _Q8_0_BLOCK_SIZE != 0:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke requires hidden size divisible by "
                f"{_Q8_0_BLOCK_SIZE}; got {columns}"
            )

        blocks_per_row = columns // _Q8_0_BLOCK_SIZE
        row_bytes = blocks_per_row * _Q8_0_BLOCK_BYTES
        logits = torch.empty(rows, dtype=torch.float32)
        payload = store.tensor_payload(tensor)
        try:
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                chunk_rows = row_end - row_start
                byte_start = row_start * row_bytes
                byte_end = row_end * row_bytes
                chunk_view = payload[byte_start:byte_end]
                try:
                    values, scales = self._decode_q8_0_output_chunk(
                        chunk_view,
                        rows=chunk_rows,
                        columns=columns,
                        blocks_per_row=blocks_per_row,
                    )
                    logits[row_start:row_end] = q8_0_matvec(
                        values,
                        scales,
                        hidden,
                        block_size=_Q8_0_BLOCK_SIZE,
                    )
                finally:
                    chunk_view.release()
        finally:
            payload.release()
        return logits

    def _decode_q8_0_output_chunk(
        self,
        payload: memoryview,
        *,
        rows: int,
        columns: int,
        blocks_per_row: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = _torch_from_readonly_buffer(payload, dtype=torch.uint8).reshape(
            rows * blocks_per_row,
            _Q8_0_BLOCK_BYTES,
        )
        scale_bytes = raw[:, :2].contiguous()
        values = raw[:, 2:].contiguous().view(torch.int8).reshape(rows, columns)
        scales = scale_bytes.view(torch.float16).to(torch.float32).reshape(
            rows,
            blocks_per_row,
        )
        return values, scales


def _torch_from_readonly_buffer(
    payload: memoryview,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_READONLY_BUFFER_WARNING)
        return torch.frombuffer(payload, dtype=dtype)
