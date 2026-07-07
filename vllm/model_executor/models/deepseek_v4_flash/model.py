# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.kernels.triton.deepseek_v4_flash.q8_linear import q8_0_raw_linear

from .block import (
    DeepSeekV4FlashCompressedLayerReferenceRunner,
    DeepSeekV4FlashSlidingLayerReferenceRunner,
)
from .compressed_kv import DeepSeekV4CompressedKVCache, DeepSeekV4PagedKVCache
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
    _stage_expert_token_table,
    deepseek_v4_flash_compressed_layer_forward,
    deepseek_v4_flash_compressed_layer_forward_batched,
    deepseek_v4_flash_sliding_layer_forward,
    deepseek_v4_flash_sliding_layer_forward_batched,
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
_OUTPUT_PROJECTION_CHUNK_ROWS = 16384
_READONLY_BUFFER_WARNING = "The given buffer is not writable"

if TYPE_CHECKING:
    from .decode_graph import DeepSeekV4FlashDecodeGraph


@dataclass
class DeepSeekV4FlashGreedyKernelSession:
    state: DeepSeekV4FlashGPURequestState
    output_ids: torch.Tensor
    input_token_count: int
    token_id_tensor: torch.Tensor
    token_id: int | None
    device: torch.device
    decode_graphs: dict[int, DeepSeekV4FlashDecodeGraph]
    decode_graph_window_shapes: dict[int, dict[int, tuple[int, ...]]]


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
        self._gpu_request_seq = 0
        self._gpu_kv_pools: dict[
            tuple[int, int, torch.dtype, str, int], DeepSeekV4PagedKVCache
        ] = {}
        self._gpu_request_states: dict[str, DeepSeekV4FlashGPURequestState] = {}
        self._gpu_request_states_high_water = 0
        self._gpu_sessions: dict[str, DeepSeekV4FlashGreedyKernelSession] = {}
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
        self._gpu_kv_pools.clear()
        self._gpu_request_states.clear()
        self._gpu_request_states_high_water = 0
        self._gpu_sessions.clear()
        self._gpu_request_seq = 0

    def close(self) -> None:
        if self.weight_store is not None:
            self.weight_store.close()
            self.weight_store = None
        self._gpu_weight_stager = None
        self._gpu_weight_stager_store_id = None
        self._gpu_weight_stager_device = None
        self._gpu_kv_pools.clear()
        self._gpu_request_states.clear()
        self._gpu_request_states_high_water = 0
        self._gpu_sessions.clear()
        self._gpu_request_seq = 0

    def _new_gpu_request_state(
        self,
        *,
        context_length: int,
        device: torch.device,
        max_requests: int = 1,
        request_id: str | None = None,
    ) -> DeepSeekV4FlashGPURequestState:
        config = DeepSeekV4FlashGPUCacheConfig(
            context_length=context_length,
            hidden_size=self.shape.hidden_size,
            batch_size=1,
            kv_width=self.shape.head_dim,
            device=device,
        )
        key = (
            context_length,
            self.shape.head_dim,
            config.dtype,
            str(device),
            int(max_requests),
        )
        pool = self._gpu_kv_pools.get(key)
        if pool is None:
            pool = DeepSeekV4PagedKVCache(
                context_length=context_length,
                hidden_size=self.shape.head_dim,
                raw_window=DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
                num_layers=DEEPSEEK_V4_FLASH_SHAPE.num_layers,
                dtype=config.dtype,
                device=device,
                max_requests=max_requests,
            )
            self._gpu_kv_pools[key] = pool
        if request_id is None:
            request_id = f"gpu-{self._gpu_request_seq}"
            self._gpu_request_seq += 1
        return DeepSeekV4FlashGPURequestState(
            config,
            kv_cache=pool,
            request_id=request_id,
        )

    def raw_block_size(self) -> int:
        return 16

    def num_raw_blocks_per_seq(self) -> int:
        raw_window = DEEPSEEK_V4_FLASH_SHAPE.sliding_window
        block_size = self.raw_block_size()
        return (raw_window + block_size - 1) // block_size

    def num_layers(self) -> int:
        return int(self.shape.num_layers)

    def device(self) -> torch.device:
        if self._gpu_weight_stager_device is not None:
            return self._gpu_weight_stager_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensure_request_state(
        self,
        *,
        request_id: str,
        context_length: int,
        device: torch.device,
        max_active_requests: int,
    ) -> DeepSeekV4FlashGPURequestState:
        existing = self._gpu_request_states.get(request_id)
        if existing is not None:
            return existing
        state = self._new_gpu_request_state(
            context_length=context_length,
            device=device,
            max_requests=max_active_requests,
            request_id=request_id,
        )
        self._gpu_request_states[request_id] = state
        self._gpu_request_states_high_water = max(
            self._gpu_request_states_high_water,
            len(self._gpu_request_states),
        )
        return state

    def ensure_request_capacity(self, request_id: str, token_idx: int) -> None:
        self.get_request_state(request_id).require_capacity(token_idx)

    def get_request_state(self, request_id: str) -> DeepSeekV4FlashGPURequestState:
        return self._gpu_request_states[request_id]

    def free_request_state(self, request_id: str) -> None:
        state = self._gpu_request_states.pop(request_id, None)
        if state is not None:
            state.reset()

    def kv_stats(self) -> dict[str, Any]:
        return {
            "active_requests": len(self._gpu_request_states),
            "active_requests_high_water": self._gpu_request_states_high_water,
            "num_pools": len(self._gpu_kv_pools),
        }

    def prefill_request(
        self,
        request_id: str,
        input_ids: list[int],
        max_tokens: int,
    ) -> int:
        device = self.device()
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        session = self.prefill_greedy_kernel(
            input_tensor,
            max_tokens=max_tokens,
            request_id=request_id,
        )
        self._gpu_sessions[request_id] = session
        self._gpu_request_states[request_id] = session.state
        self._gpu_request_states_high_water = max(
            self._gpu_request_states_high_water,
            len(self._gpu_request_states),
        )
        return self.decode_single_token(request_id)

    def decode_single_token(self, request_id: str) -> int:
        session = self._gpu_sessions[request_id]
        output_ids, _elapsed = self.decode_greedy_kernel(
            session,
            max_tokens=1,
            use_graph=False,
            reset_state=False,
        )
        token = int(output_ids[session.input_token_count].detach().cpu().item())
        session.input_token_count += 1
        return token

    def decode_tokens_batch(self, request_ids: list[str]) -> torch.Tensor:
        if not request_ids:
            return torch.empty((0,), dtype=torch.long, device=self.device())
        sessions = [self._gpu_sessions[request_id] for request_id in request_ids]
        device = sessions[0].device
        token_id_tensor = torch.stack(
            [
                session.token_id_tensor.to(device=device, dtype=torch.long)
                for session in sessions
            ]
        ).reshape(len(sessions))
        next_tokens = self._forward_kernel_token_step_batched(
            token_id_tensor=token_id_tensor,
            states=[session.state for session in sessions],
            token_indices=[session.state.token_position for session in sessions],
            device=device,
        )
        for session, token in zip(sessions, next_tokens, strict=True):
            session.output_ids[session.input_token_count] = token
            session.input_token_count += 1
            session.token_id_tensor = token.reshape(())
            session.token_id = int(token.detach().cpu().item())
        return next_tokens.to(device=self.device(), dtype=torch.long)

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
        cache_admission_policy = getattr(stager, "cache_admission_policy", None)
        cacheable_expert_ids = tuple(
            expert_id
            for expert_id in dict.fromkeys(expert_ids)
            if cache_admission_policy is None
            or cache_admission_policy.should_cache_grouped_expert(
                layer_idx=layer_idx,
                expert_id=expert_id,
            )
        )
        if not cacheable_expert_ids:
            return
        try:
            stager.prefetch_grouped_experts(
                layer,
                DeepSeekV4FlashExpertPrefetchRequest(
                    layer_idx=layer_idx,
                    expert_ids=cacheable_expert_ids,
                ),
            )
        except Exception:
            self._deepseek_profiler.add_counter("deepseek_prefetch_failures", 1)

    def _likely_hash_routed_expert_ids_for_token(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        layer: Any,
        *,
        token_id: int | None,
    ) -> tuple[int, ...]:
        if token_id is None:
            return ()
        expert_token_table = getattr(layer, "expert_token_to_expert_ids", None)
        if expert_token_table is None:
            return ()
        table = stager.store.tensor_to_torch(expert_token_table, dtype=torch.int32)
        if table.ndim != 2:
            raise ValueError(f"expert token table must be 2-D; got {table.ndim}-D")
        if table.is_cuda:
            return ()
        if token_id < 0 or token_id >= table.shape[1]:
            raise ValueError(
                f"token_id out of range: {token_id}; expected [0, {table.shape[1]})"
            )
        expert_ids = {
            int(expert_id)
            for expert_id in table[:, token_id].tolist()
            if int(expert_id) >= 0
        }
        return tuple(sorted(expert_ids))

    def _schedule_next_layer_expert_prefetch(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        next_layer: Any,
        *,
        token_id: int | None,
    ) -> None:
        grouped_experts = getattr(next_layer, "grouped_experts", None)
        if grouped_experts is None:
            return
        expert_ids = self._likely_hash_routed_expert_ids_for_token(
            stager,
            next_layer,
            token_id=token_id,
        )
        if not expert_ids:
            return
        self._prefetch_grouped_experts_best_effort(
            stager,
            grouped_experts,
            expert_ids,
            layer_idx=next_layer.layer_index,
        )

    def _schedule_next_layer_expert_prefetch_async(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        next_layer: Any,
        *,
        token_id: int | None,
    ) -> torch.cuda.Event | None:
        """Enqueue an async prefetch for the next layer and return its event.

        Returns None if there is nothing to prefetch or prefetch fails.
        """
        grouped_experts = getattr(next_layer, "grouped_experts", None)
        if grouped_experts is None:
            return None
        expert_ids = self._likely_hash_routed_expert_ids_for_token(
            stager,
            next_layer,
            token_id=token_id,
        )
        if not expert_ids:
            return None
        cache_admission_policy = getattr(stager, "cache_admission_policy", None)
        cacheable_expert_ids = tuple(
            expert_id
            for expert_id in dict.fromkeys(expert_ids)
            if cache_admission_policy is None
            or cache_admission_policy.should_cache_grouped_expert(
                layer_idx=next_layer.layer_index,
                expert_id=expert_id,
            )
        )
        if not cacheable_expert_ids:
            return None
        try:
            return stager.prefetch_grouped_experts_async(
                grouped_experts,
                DeepSeekV4FlashExpertPrefetchRequest(
                    layer_idx=next_layer.layer_index,
                    expert_ids=cacheable_expert_ids,
                ),
            )
        except Exception:
            self._deepseek_profiler.add_counter("deepseek_prefetch_failures", 1)
            return None

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
        state = self._new_gpu_request_state(
            context_length=self._kernel_context_length(),
            device=device,
        )
        try:
            return self._forward_kernel_token_step(
                token_id=token_id,
                state=state,
                token_idx=state.token_position,
                device=device,
            )
        finally:
            state.reset()

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
        compressed_counts_by_layer = self._compute_graph_compressed_counts(
            token_idx=token_idx,
        )
        store, stager, hidden = self._forward_kernel_token_hidden(
            token_id=token_id,
            state=state,
            token_idx=token_idx,
            device=device,
            compressed_counts_by_layer=compressed_counts_by_layer,
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
        advance_state: bool = True,
        kv_rows_by_layer: dict[int, torch.Tensor] | None = None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] | None = None,
        compressed_counts_by_layer: dict[int, int] | None = None,
    ) -> torch.Tensor:
        store, stager, hidden = self._forward_kernel_token_hidden_token_tensor(
            token_id_tensor=token_id_tensor,
            state=state,
            token_idx=token_idx,
            device=device,
            kv_rows_by_layer=kv_rows_by_layer,
            extra_kv_rows_by_layer=extra_kv_rows_by_layer,
            compressed_counts_by_layer=compressed_counts_by_layer,
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
        if advance_state:
            state.advance_token()
        return next_token.to(device=device, dtype=torch.long).reshape(())

    def _forward_kernel_token_hidden(
        self,
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
        compressed_counts_by_layer: dict[int, int] | None = None,
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
        layers = list(layers)
        async_prefetch_enabled = self._deepseek_async_prefetch_enabled()
        pending_prefetch_event: torch.cuda.Event | None = None
        for layer_offset, layer in enumerate(layers):
            if pending_prefetch_event is not None:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_prefetch_wait",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    stager.wait_for_prefetch(pending_prefetch_event)
                pending_prefetch_event = None
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
                        state=state,
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
                        compressed_count=compressed_counts_by_layer.get(
                            layer.layer_index
                        )
                        if compressed_counts_by_layer is not None
                        else None,
                    )
            if layer_offset + 1 < len(layers):
                next_layer = layers[layer_offset + 1]
                if async_prefetch_enabled:
                    pending_prefetch_event = (
                        self._schedule_next_layer_expert_prefetch_async(
                            stager,
                            next_layer,
                            token_id=token_id,
                        )
                    )
                else:
                    self._schedule_next_layer_expert_prefetch(
                        stager,
                        next_layer,
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
        kv_rows_by_layer: dict[int, torch.Tensor] | None = None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] | None = None,
        compressed_counts_by_layer: dict[int, int] | None = None,
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
        layers = list(layers)
        for layer_offset, layer in enumerate(layers):
            layer_kv_rows = (
                kv_rows_by_layer.get(layer.layer_index)
                if kv_rows_by_layer is not None
                else None
            )
            layer_extra_kv_rows = (
                extra_kv_rows_by_layer.get(layer.layer_index)
                if extra_kv_rows_by_layer is not None
                else None
            )
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
                        state=state,
                        token_idx=token_idx,
                        token_id_tensor=token_id_tensor,
                        kv_rows=layer_kv_rows,
                        extra_kv_rows=layer_extra_kv_rows,
                        kv_rows_by_layer=kv_rows_by_layer,
                        extra_kv_rows_by_layer=extra_kv_rows_by_layer,
                    )
            else:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_compressed",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    compressed_count = (
                        compressed_counts_by_layer.get(layer.layer_index)
                        if compressed_counts_by_layer is not None
                        else None
                    )
                    hidden = deepseek_v4_flash_compressed_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        state=state,
                        token_idx=token_idx,
                        token_id_tensor=token_id_tensor,
                        kv_rows=layer_kv_rows,
                        extra_kv_rows=layer_extra_kv_rows,
                        kv_rows_by_layer=kv_rows_by_layer,
                        extra_kv_rows_by_layer=extra_kv_rows_by_layer,
                        compressed_count=compressed_count,
                    )
            if layer_offset + 1 < len(layers):
                self._schedule_next_layer_expert_prefetch(
                    stager,
                    layers[layer_offset + 1],
                    token_id=None,
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
        use_graph: bool = False,
    ) -> torch.Tensor:
        with self._deepseek_profiler.section(
            "generate_greedy_kernel",
            input_tokens=int(input_ids.numel()),
            max_tokens=max_tokens,
            use_graph=use_graph,
        ):
            output_ids, _token_elapsed_ms = self._generate_greedy_kernel_impl(
                input_ids,
                max_tokens=max_tokens,
                record_token_times=False,
                use_graph=use_graph,
            )
            return output_ids

    def generate_greedy_kernel_timed(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int = 1,
        use_graph: bool = False,
    ) -> tuple[torch.Tensor, list[float]]:
        return self._generate_greedy_kernel_impl(
            input_ids,
            max_tokens=max_tokens,
            record_token_times=True,
            use_graph=use_graph,
        )

    def generate_greedy_kernel_batched(
        self,
        input_ids_list: list[torch.Tensor],
        max_tokens: int,
        *,
        record_token_times: bool = False,
    ) -> list[torch.Tensor]:
        """Greedy decode a batch of requests using the batched decode kernel.

        Prefill is performed per-request with the single-slot kernel. Decode
        steps run batched through ``_forward_kernel_token_step_batched``.
        Graph capture is never used, even when the backend supports it.

        ``record_token_times`` is accepted for API symmetry with
        ``generate_greedy_kernel`` but is currently ignored.
        """
        with self._deepseek_profiler.section(
            "generate_greedy_kernel_batched",
            batch_size=len(input_ids_list),
            max_tokens=max_tokens,
            record_token_times=record_token_times,
        ):
            return self._generate_greedy_kernel_batched_impl(
                input_ids_list,
                max_tokens=max_tokens,
            )

    def _generate_greedy_kernel_batched_impl(
        self,
        input_ids_list: list[torch.Tensor],
        *,
        max_tokens: int,
    ) -> list[torch.Tensor]:
        self._require_weight_store()
        device = self._validate_generate_greedy_kernel_batched_input(
            input_ids_list,
            max_tokens=max_tokens,
        )
        self.gpu_backend.require_ready()
        eos_token_id = self._eos_token_id()
        context_length = self._kernel_context_length()

        output_ids_list: list[torch.Tensor] = []
        states: list[DeepSeekV4FlashGPURequestState] = []
        input_lengths: list[int] = []

        for input_ids in input_ids_list:
            input_ids_cuda = input_ids.to(device=device, dtype=torch.long)
            input_len = int(input_ids_cuda.numel())
            input_lengths.append(input_len)

            output_ids = torch.empty(
                (input_len + max_tokens,),
                dtype=torch.long,
                device=device,
            )
            output_ids[:input_len] = input_ids_cuda
            output_ids_list.append(output_ids)

            state = self._new_gpu_request_state(
                context_length=context_length,
                device=device,
                max_requests=len(input_ids_list),
            )
            states.append(state)

            # Prefill all prompt tokens except the last one. The last prompt
            # token is consumed by the first batched decode step.
            prompt_prefix_ids = input_ids_cuda[:-1].detach().cpu().tolist()
            for token_id in prompt_prefix_ids:
                self._forward_kernel_token_step_token_id(
                    token_id=int(token_id),
                    state=state,
                    token_idx=state.token_position,
                    device=device,
                )

        try:
            active_indices = list(range(len(states)))
            for step in range(max_tokens):
                if not active_indices:
                    break

                token_id_tensor = torch.stack(
                    [
                        output_ids_list[idx][input_lengths[idx] - 1 + step]
                        for idx in active_indices
                    ]
                )
                token_indices = [states[idx].token_position for idx in active_indices]
                active_states = [states[idx] for idx in active_indices]

                next_tokens = self._forward_kernel_token_step_batched(
                    token_id_tensor=token_id_tensor,
                    states=active_states,
                    token_indices=token_indices,
                    device=device,
                )

                new_active: list[int] = []
                for offset, idx in enumerate(active_indices):
                    generated_token = next_tokens[offset]
                    output_ids_list[idx][input_lengths[idx] + step] = generated_token
                    if eos_token_id is not None:
                        token_id_int = int(generated_token.item())
                        if token_id_int == eos_token_id:
                            continue
                    new_active.append(idx)
                active_indices = new_active
        finally:
            for state in states:
                state.reset()

        # Trim each output to the actual number of generated tokens, matching
        # the single-slot greedy path which returns a slice up to EOS.
        results: list[torch.Tensor] = []
        for idx, output_ids in enumerate(output_ids_list):
            actual_end = input_lengths[idx] + max_tokens
            generated_slice = output_ids[input_lengths[idx] : actual_end]
            if eos_token_id is not None:
                eos_positions = (generated_slice == eos_token_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() > 0:
                    actual_end = input_lengths[idx] + int(eos_positions[0].item()) + 1
            results.append(output_ids[:actual_end])
        return results

    def _forward_kernel_token_step_batched(
        self,
        *,
        token_id_tensor: torch.Tensor,
        states: list[DeepSeekV4FlashGPURequestState],
        token_indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        store, stager, hidden = self._forward_kernel_token_hidden_batched(
            token_id_tensor=token_id_tensor,
            states=states,
            token_indices=token_indices,
            device=device,
        )
        with self._deepseek_profiler.section(
            "output_projection",
            token_idx=token_indices[0] if token_indices else -1,
        ):
            next_tokens: list[torch.Tensor] = []
            for b in range(hidden.shape[0]):
                streams = self._kernel_output_streams(hidden[b])
                next_tokens.append(
                    self._output_token_argmax_chunked_cuda(
                        store,
                        stager=stager,
                        streams=streams,
                        device=device,
                    )
                )
        result = torch.stack(next_tokens).to(device=device, dtype=torch.long)
        for state in states:
            state.advance_token()
        return result

    def _forward_kernel_token_hidden_batched(
        self,
        *,
        token_id_tensor: torch.Tensor,
        states: list[DeepSeekV4FlashGPURequestState],
        token_indices: list[int],
        device: torch.device,
    ) -> tuple[
        DeepSeekV4FlashWeightStore,
        DeepSeekV4FlashGPUWeightStager,
        torch.Tensor,
    ]:
        store = self._require_weight_store()
        self.gpu_backend.require_ready()
        if token_id_tensor.ndim != 1:
            raise ValueError(
                f"batched token id tensor must be 1-D; got {token_id_tensor.ndim}-D"
            )
        batch = token_id_tensor.shape[0]
        if len(states) != batch or len(token_indices) != batch:
            raise ValueError(
                "states and token_indices must match token_id_tensor batch size"
            )
        if not token_id_tensor.is_cuda:
            raise ValueError("batched token id tensor must be CUDA")
        if token_id_tensor.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                "batched token id tensor must use an integer dtype; "
                f"got {token_id_tensor.dtype}"
            )

        token_id_tensor = token_id_tensor.to(device=device, dtype=torch.long)
        torch._assert_async(
            ((token_id_tensor >= 0) & (token_id_tensor < self.shape.vocab_size)).all(),
            f"batched token id is outside vocab range [0, {self.shape.vocab_size})",
        )

        for state, token_idx in zip(states, token_indices, strict=True):
            state.require_capacity(token_idx)
            if token_idx != state.token_position:
                raise ValueError(
                    "batched forward token_idx must match request state "
                    f"token_position; got token_idx={token_idx}, "
                    f"token_position={state.token_position}"
                )

        stager = self._get_gpu_weight_stager(device)
        hidden = self._stage_token_embedding_batch_cuda(
            store,
            stager,
            token_id_tensor,
            device=device,
        )

        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires layer bindings")
        layers = list(layers)
        for layer_offset, layer in enumerate(layers):
            if layer.layer_index < 2:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_sliding_batched",
                    layer_idx=layer.layer_index,
                    token_idx=token_indices[0] if token_indices else -1,
                ):
                    hidden = deepseek_v4_flash_sliding_layer_forward_batched(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        states=states,
                        token_indices=token_indices,
                        token_id_tensors=token_id_tensor,
                    )
            else:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_compressed_batched",
                    layer_idx=layer.layer_index,
                    token_idx=token_indices[0] if token_indices else -1,
                ):
                    hidden = deepseek_v4_flash_compressed_layer_forward_batched(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        states=states,
                        token_indices=token_indices,
                        token_id_tensors=token_id_tensor,
                    )
            if layer_offset + 1 < len(layers):
                self._schedule_next_layer_expert_prefetch(
                    stager,
                    layers[layer_offset + 1],
                    token_id=None,
                )

        return store, stager, hidden

    def _stage_token_embedding_batch_cuda(
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

        token_id_tensor = token_id_tensor.to(device=device, dtype=torch.long)
        embedding_matrix = stager.stage_matrix(tensor).reshape(vocab_size, hidden_size)
        hidden = torch.index_select(
            embedding_matrix,
            0,
            token_id_tensor.reshape(-1),
        )
        return hidden.to(dtype=torch.float32)

    def _validate_generate_greedy_kernel_batched_input(
        self,
        input_ids_list: list[torch.Tensor],
        *,
        max_tokens: int,
    ) -> torch.device:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive; got {max_tokens}")
        if not input_ids_list:
            raise ValueError("input_ids_list must not be empty")

        device: torch.device | None = None
        context_length = self._kernel_context_length()

        for input_ids in input_ids_list:
            if input_ids.ndim != 1:
                raise ValueError(
                    "generate_greedy_kernel_batched supports 1-D input tensors; "
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
            if input_ids.numel() == 0:
                raise ValueError(
                    "generate_greedy_kernel_batched requires at least one input token"
                )

            current_device = torch.device(input_ids.device)
            if current_device.type == "cuda":
                if device is None:
                    device = current_device
                elif device != current_device:
                    raise ValueError(
                        "generate_greedy_kernel_batched requires all CUDA input "
                        "tensors on the same device"
                    )
            elif current_device.type != "cpu":
                raise ValueError(
                    "generate_greedy_kernel_batched supports CPU or CUDA input "
                    f"tensors; got {current_device}"
                )

            total_tokens = input_ids.numel() + max_tokens
            if total_tokens > context_length:
                raise ValueError(
                    "generate_greedy_kernel_batched input plus generated tokens "
                    f"exceeds configured budget: {total_tokens} tokens > "
                    f"{context_length}"
                )

            for token_id in input_ids.detach().cpu().tolist():
                token_id_int = int(token_id)
                if token_id_int < 0 or token_id_int >= self.shape.vocab_size:
                    raise ValueError(
                        f"input token id {token_id_int} is outside vocab range "
                        f"[0, {self.shape.vocab_size})"
                    )

        if device is None:
            device = torch.device("cuda")
        return device

    def prefill_greedy_kernel(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        request_id: str | None = None,
    ) -> DeepSeekV4FlashGreedyKernelSession:
        self._require_weight_store()
        device = self._validate_generate_greedy_kernel_input(
            input_ids,
            max_tokens=max_tokens,
        )
        self.pin_hot_experts_for_input_ids(input_ids, device)
        self.gpu_backend.require_ready()
        state = (
            self._gpu_request_states.get(request_id) if request_id is not None else None
        )
        if state is None:
            state = self._new_gpu_request_state(
                context_length=self._kernel_context_length(),
                device=device,
                request_id=request_id,
            )
            if request_id is not None:
                self._gpu_request_states[request_id] = state

        input_token_count = int(input_ids.numel())
        output_ids = torch.empty(
            (input_token_count + max_tokens,),
            dtype=torch.long,
            device=device,
        )
        output_ids[:input_token_count] = input_ids.to(
            device=device,
            dtype=torch.long,
        )

        prompt_prefix_token_ids = (
            output_ids[: input_token_count - 1].detach().cpu().tolist()
        )
        for token_id in prompt_prefix_token_ids:
            self._forward_kernel_token_step_token_id(
                token_id=int(token_id),
                state=state,
                token_idx=state.token_position,
                device=device,
            )

        token_id_tensor = output_ids[input_token_count - 1].reshape(())
        eos_token_id = self._eos_token_id()
        token_id = int(token_id_tensor.item()) if eos_token_id is not None else None
        return DeepSeekV4FlashGreedyKernelSession(
            state=state,
            output_ids=output_ids,
            input_token_count=input_token_count,
            token_id_tensor=token_id_tensor,
            token_id=token_id,
            device=device,
            decode_graphs={},
            decode_graph_window_shapes={},
        )

    def decode_greedy_kernel(
        self,
        session: DeepSeekV4FlashGreedyKernelSession,
        *,
        max_tokens: int,
        record_token_times: bool = False,
        use_graph: bool = False,
        reset_state: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        state = session.state
        output_ids = session.output_ids
        input_token_count = session.input_token_count
        token_id_tensor = session.token_id_tensor
        token_id = session.token_id
        device = session.device
        decode_graphs = session.decode_graphs
        decode_graph_window_shapes = session.decode_graph_window_shapes
        eos_token_id = self._eos_token_id()

        token_elapsed_ms: list[float] = []
        for generated_idx in range(max_tokens):
            start_event: torch.cuda.Event | None = None
            end_event: torch.cuda.Event | None = None
            if record_token_times:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            current_token_idx = state.token_position
            can_capture_graph = (
                use_graph
                and self._model_layers_all_hash_routed()
                and not self._decode_step_is_emit_boundary(current_token_idx)
            )
            can_replay_graph = (
                can_capture_graph
                and current_token_idx in decode_graphs
                and self._decode_kv_window_shapes_match(
                    state=state,
                    token_idx=current_token_idx,
                    captured_shapes=decode_graph_window_shapes.get(current_token_idx),
                )
            )
            if can_replay_graph:
                compressed_counts_by_layer = self._compute_graph_compressed_counts(
                    token_idx=current_token_idx,
                )
                decode_graphs[current_token_idx].prepare_replay(
                    self,
                    state=state,
                    token_idx=current_token_idx,
                    token_id_tensor=token_id_tensor,
                    device=device,
                    compressed_counts_by_layer=compressed_counts_by_layer,
                )
                token_id_tensor = decode_graphs[current_token_idx].replay(
                    token_id_tensor,
                    compressed_counts_by_layer=compressed_counts_by_layer,
                )
                state.advance_token()
            elif can_capture_graph:
                kv_rows_by_layer, extra_kv_rows_by_layer = (
                    self._materialize_decode_kv_windows(
                        state=state,
                        token_idx=current_token_idx,
                    )
                )
                compressed_counts_by_layer = self._compute_graph_compressed_counts(
                    token_idx=current_token_idx,
                )
                decode_graphs[current_token_idx] = self._capture_decode_graph(
                    state=state,
                    token_idx=current_token_idx,
                    device=device,
                    kv_rows_by_layer=kv_rows_by_layer,
                    extra_kv_rows_by_layer=extra_kv_rows_by_layer,
                    compressed_counts_by_layer=compressed_counts_by_layer,
                )
                decode_graph_window_shapes[current_token_idx] = (
                    self._decode_kv_window_shapes(
                        state=state,
                        token_idx=current_token_idx,
                    )
                )
                decode_graphs[current_token_idx].prepare_replay(
                    self,
                    state=state,
                    token_idx=current_token_idx,
                    token_id_tensor=token_id_tensor,
                    device=device,
                    compressed_counts_by_layer=compressed_counts_by_layer,
                )
                token_id_tensor = decode_graphs[current_token_idx].replay(
                    token_id_tensor,
                    compressed_counts_by_layer=compressed_counts_by_layer,
                )
                state.advance_token()
            elif token_id is None:
                token_id_tensor = self._forward_kernel_token_step_token_tensor(
                    token_id_tensor=token_id_tensor,
                    state=state,
                    token_idx=state.token_position,
                    device=device,
                )
            else:
                token_id_tensor = self._forward_kernel_token_step_token_id(
                    token_id=token_id,
                    state=state,
                    token_idx=state.token_position,
                    device=device,
                )
            if record_token_times:
                if start_event is None or end_event is None:
                    raise AssertionError("CUDA timing events were not initialized")
                end_event.record()
                torch.cuda.synchronize()
                token_elapsed_ms.append(float(start_event.elapsed_time(end_event)))
            output_ids[input_token_count + generated_idx] = token_id_tensor
            if eos_token_id is not None:
                token_id = int(token_id_tensor.item())
                if token_id == eos_token_id:
                    if reset_state:
                        state.reset()
                    else:
                        session.token_id_tensor = token_id_tensor
                        session.token_id = token_id
                    return (
                        output_ids[: input_token_count + generated_idx + 1],
                        token_elapsed_ms,
                    )

        if reset_state:
            state.reset()
        else:
            session.token_id_tensor = token_id_tensor
            session.token_id = int(token_id_tensor.detach().cpu().item())
        return output_ids, token_elapsed_ms

    def _generate_greedy_kernel_impl(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        record_token_times: bool,
        use_graph: bool,
    ) -> tuple[torch.Tensor, list[float]]:
        session = self.prefill_greedy_kernel(
            input_ids,
            max_tokens=max_tokens,
        )
        return self.decode_greedy_kernel(
            session,
            max_tokens=max_tokens,
            record_token_times=record_token_times,
            use_graph=use_graph,
        )

    def _eos_token_id(self) -> int | None:
        store = self._require_weight_store()
        model = getattr(store, "model", None)
        metadata = getattr(model, "metadata", {})
        eos_token_id = metadata.get("tokenizer.ggml.eos_token_id")
        return None if eos_token_id is None else int(eos_token_id)

    def _decode_step_is_emit_boundary(self, token_idx: int) -> bool:
        """Return True if any compressed layer emits at ``token_idx``."""
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return False
        for layer in layers:
            ratio = layer_compress_ratio(layer.layer_index)
            if ratio > 0 and token_idx % ratio == ratio - 1:
                return True
        return False

    def _model_layers_all_hash_routed(self) -> bool:
        """Return True iff every layer uses hash-routed experts.

        CUDA/HIP graph capture currently depends on expert payloads being
        stageable outside the captured region via the static token-to-expert
        table. Top-k routed layers select experts from the hidden state inside
        the graph and cannot be captured safely yet.
        """
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if not layers:
            return True
        for layer in layers:
            if layer.grouped_experts is None:
                continue
            if layer.expert_token_to_expert_ids is None:
                return False
        return True

    def _compute_graph_compressed_counts(
        self,
        *,
        token_idx: int,
    ) -> dict[int, int]:
        """Return per-layer compressed row counts for a graph capture/replay.

        The count matches the number of compressed rows that exist just before
        the decode step at ``token_idx``; this avoids a CPU sync inside the
        captured region.
        """
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return {}
        counts: dict[int, int] = {}
        for layer in layers:
            ratio = layer_compress_ratio(layer.layer_index)
            if ratio > 0:
                counts[layer.layer_index] = (token_idx + 1) // ratio
        return counts

    def _capture_decode_graph(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
        kv_rows_by_layer: dict[int, torch.Tensor] | None = None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] | None = None,
        compressed_counts_by_layer: dict[int, int] | None = None,
    ) -> DeepSeekV4FlashDecodeGraph:
        """Capture a CUDA/HIP graph for one steady-state decode step."""
        from .decode_graph import DeepSeekV4FlashDecodeGraph

        return DeepSeekV4FlashDecodeGraph.capture(
            self,
            state=state,
            token_idx=token_idx,
            device=device,
            kv_rows_by_layer=kv_rows_by_layer,
            extra_kv_rows_by_layer=extra_kv_rows_by_layer,
            compressed_counts_by_layer=compressed_counts_by_layer,
        )

    def _materialize_decode_kv_windows(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor | None]]:
        """Return explicit KV windows for all layers at ``token_idx``.

        Sliding layers return their raw sliding window as ``kv_rows``.
        Compressed layers leave ``kv_rows`` as ``None`` (the raw window is read
        inside the graph) and return their compressed extra rows as
        ``extra_kv_rows`` so the graph sees stable inputs for every layer.

        Compressed layers with an indexer return ``None`` for ``extra_kv_rows``;
        the indexer selection depends on the query and must run inside the
        graph to match the reference path.
        """
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return {}, {}
        kv_rows_by_layer: dict[int, torch.Tensor] = {}
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] = {}
        for layer in layers:
            ratio = layer_compress_ratio(layer.layer_index)
            kv_rows_by_layer[layer.layer_index] = state.raw_kv_window(
                layer.layer_index,
                token_idx,
                self.shape.sliding_window,
            )
            if ratio != 0:
                extra_kv_rows_by_layer[layer.layer_index] = (
                    self._materialize_compressed_extra_rows(
                        state=state,
                        layer=layer,
                        token_idx=token_idx,
                    )
                )
        return kv_rows_by_layer, extra_kv_rows_by_layer

    def _stage_graph_moe_payloads(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_id: int,
        device: torch.device,
    ) -> None:
        """Stage the MoE payloads needed for one graph replay step.

        The returned payload tensors are stable cached objects; their bytes are
        copied in place before each replay so the captured graph sees the
        current token's experts without running CPU-side staging inside the
        graph.
        """
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return
        stager = self._get_gpu_weight_stager(device)
        for layer in layers:
            if layer.expert_token_to_expert_ids is None:
                continue
            grouped = layer.grouped_experts
            if grouped is None:
                continue
            table = _stage_expert_token_table(
                stager,
                layer.expert_token_to_expert_ids,
                device=torch.device("cpu"),
            )
            if token_id < 0 or token_id >= table.shape[1]:
                raise ValueError(
                    f"graph token_id out of range: {token_id}; "
                    f"expected [0, {table.shape[1]})"
                )
            expert_ids = table[:, token_id].to(torch.int64)
            payloads = self.gpu_backend.stage_selected_expert_payloads(
                stager,
                grouped,
                expert_ids,
                layer_idx=layer.layer_index,
            )
            state.set_graph_moe_payloads(layer.layer_index, payloads)

    def _materialize_compressed_extra_rows(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        layer: Any,
        token_idx: int,
    ) -> torch.Tensor | None:
        """Return the compressed extra rows to feed into a graph step.

        Layers without an indexer use all prior compressed rows, which is
        stable across decode steps and safe to materialize once per graph.
        Layers with an indexer select rows based on the current query, so
        returning ``None`` lets ``_run_real_sliding_attention`` compute the
        indexer selection inside the graph and match the reference path.
        """
        del token_idx
        if layer.indexer is not None:
            return None
        return state.compressed_kv_cache.read_compressed(layer.layer_index)

    def _decode_kv_window_shapes(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
    ) -> dict[int, tuple[int, ...] | None]:
        """Return the shapes of KV windows for all layers at ``token_idx``."""
        store = self._require_weight_store()
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return {}
        shapes: dict[int, tuple[int, ...] | None] = {}
        for layer in layers:
            ratio = layer_compress_ratio(layer.layer_index)
            shapes[layer.layer_index] = tuple(
                state.raw_kv_window(
                    layer.layer_index,
                    token_idx,
                    self.shape.sliding_window,
                ).shape
            )
            if ratio != 0 and layer.indexer is None:
                shapes[layer.layer_index] = shapes[layer.layer_index] + tuple(
                    state.compressed_kv_cache.read_compressed(layer.layer_index).shape
                )
        return shapes

    def _decode_kv_window_shapes_match(
        self,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        captured_shapes: dict[int, tuple[int, ...] | None] | None,
    ) -> bool:
        """Return True if the current decode KV window shapes match capture."""
        if captured_shapes is None:
            return False
        current_shapes = self._decode_kv_window_shapes(
            state=state,
            token_idx=token_idx,
        )
        return current_shapes == captured_shapes

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
                max_staged_bytes=(
                    None
                    if self._deepseek_full_resident_enabled()
                    else self._gpu_staging_budget_bytes()
                ),
            )
            if self._deepseek_full_resident_enabled():
                self._gpu_weight_stager.enable_full_resident_mode()
            self._gpu_weight_stager_store_id = store_id
            self._gpu_weight_stager_device = device
        self._gpu_weight_stager.profiler = self._deepseek_profiler
        self.gpu_backend.profiler = self._deepseek_profiler
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
        base = max(
            0,
            self.runtime_budget.available_headroom_bytes
            - self.runtime_budget.min_system_headroom_bytes,
        )
        extra_gb_str = os.environ.get(
            "FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB",
            "0",
        )
        try:
            extra_gb = float(extra_gb_str)
        except ValueError:
            warnings.warn(
                f"Ignoring malformed FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB "
                f"value {extra_gb_str!r}; using 0",
                stacklevel=2,
            )
            extra_gb = 0.0
        if extra_gb <= 0:
            return base
        extra_bytes = int(extra_gb * 1024 * 1024 * 1024)
        return min(
            base + extra_bytes,
            self.runtime_budget.available_headroom_bytes,
        )

    @staticmethod
    def _deepseek_full_resident_enabled() -> bool:
        return (
            os.environ.get(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT",
                "0",
            )
            == "1"
        )

    @staticmethod
    def _deepseek_async_prefetch_enabled() -> bool:
        return (
            os.environ.get(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH",
                "0",
            )
            == "1"
        )

    def pin_hot_experts_for_input_ids(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Pin experts mapped from the prompt tokens for all hash-routed layers."""
        if (
            os.environ.get(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS",
                "1",
            )
            != "1"
        ):
            return
        store = self._require_weight_store()
        stager = self._get_gpu_weight_stager(device)
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            return
        token_ids = set(int(t) for t in input_ids.detach().cpu().tolist())
        for layer in layers:
            expert_token_table = getattr(layer, "expert_token_to_expert_ids", None)
            if expert_token_table is None:
                continue
            if isinstance(expert_token_table, DeepSeekV4FlashTensor):
                table = _stage_expert_token_table(
                    stager, expert_token_table, device=stager.device
                )
            elif isinstance(expert_token_table, torch.Tensor):
                table = expert_token_table
            else:
                continue
            if table.ndim != 2:
                continue
            table_device = stager.device if table.is_cuda else torch.device("cpu")
            staged_table = table.to(table_device)
            for token_id in token_ids:
                if token_id < 0 or token_id >= staged_table.shape[1]:
                    continue
                for expert_id in staged_table[:, token_id].tolist():
                    stager.pin_grouped_expert(
                        layer.layer_index,
                        int(expert_id),
                    )

    def gpu_staging_memory_stats(self) -> dict[str, int | None]:
        stager = self._gpu_weight_stager
        if stager is None:
            return {
                "staged_bytes": 0,
                "max_staged_bytes": self._gpu_staging_budget_bytes(),
                "full_resident_enabled": int(self._deepseek_full_resident_enabled()),
                "dynamic_entries": 0,
                "grouped_entries": 0,
            }
        return stager.memory_stats()

    def prepare_deepseek_hot_experts(
        self,
        *,
        device: torch.device,
        max_layers: int = 4,
        experts_per_layer: int = 2,
    ) -> dict[str, int | None]:
        store = self._require_weight_store()
        stager = self._get_gpu_weight_stager(device)
        if max_layers <= 0 or experts_per_layer <= 0:
            return stager.memory_stats()

        stage_payload = getattr(stager, "stage_grouped_expert_payload", None)
        if stage_payload is None:
            return stager.memory_stats()

        layers = sorted(
            (
                layer
                for layer in getattr(store.bindings, "layers", ())
                if getattr(layer, "grouped_experts", None) is not None
            ),
            key=lambda layer: layer.layer_index,
        )
        pinned_experts = self._discover_stager_pinned_experts(stager)
        pinned_by_layer: dict[int, list[int]] = {}
        for layer_idx, expert_id in pinned_experts:
            pinned_by_layer.setdefault(layer_idx, []).append(expert_id)
        for expert_ids in pinned_by_layer.values():
            expert_ids.sort()

        prepared_layers = 0
        for layer in layers:
            grouped_experts = layer.grouped_experts
            if grouped_experts is None:
                continue
            if pinned_by_layer:
                expert_ids = pinned_by_layer.get(layer.layer_index, [])
                if not expert_ids:
                    continue
                expert_ids = expert_ids[:experts_per_layer]
            else:
                expert_count = self._grouped_expert_count(grouped_experts)
                expert_ids = list(range(min(experts_per_layer, expert_count)))
            if not expert_ids:
                continue

            for expert_id in expert_ids:
                for tensor in (
                    getattr(grouped_experts, "gate", None),
                    getattr(grouped_experts, "up", None),
                    getattr(grouped_experts, "down", None),
                ):
                    if tensor is None:
                        continue
                    try:
                        stage_payload(
                            tensor,
                            expert_id,
                            layer_idx=layer.layer_index,
                        )
                    except (AttributeError, KeyError, NotImplementedError):
                        continue
            prepared_layers += 1
            if prepared_layers >= max_layers:
                break

        return stager.memory_stats()

    @staticmethod
    def _discover_stager_pinned_experts(
        stager: DeepSeekV4FlashGPUWeightStager,
    ) -> tuple[tuple[int, int], ...]:
        pinned_experts: set[tuple[int, int]] = set()
        hot_policy = getattr(stager, "hot_expert_policy", None)
        pinned_experts.update(getattr(hot_policy, "pinned_experts", frozenset()))
        pinned_experts.update(getattr(stager, "_manual_pinned_experts", frozenset()))
        return tuple(sorted(pinned_experts))

    @staticmethod
    def _grouped_expert_count(
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    ) -> int:
        counts = [
            tensor.dims[2]
            for tensor in (
                grouped_experts.gate,
                grouped_experts.up,
                grouped_experts.down,
            )
            if len(tensor.dims) == 3
        ]
        return min(counts, default=0)

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

    def warm_decode_static_weights(self, device: torch.device) -> None:
        store = self._require_weight_store()
        output_head = getattr(store.bindings, "output_head", None)
        if output_head is None:
            raise RuntimeError("DeepSeek V4 Flash output.weight binding is required")
        stager = self._get_gpu_weight_stager(device)
        stager.warm_static_decode_weights(output_weight=output_head)

    def warm_decode_token_experts(self, input_ids: torch.Tensor) -> None:
        store = self._require_weight_store()
        stager = self._get_gpu_weight_stager(input_ids.device)
        token_id = int(input_ids[-1].detach().cpu().item())
        for layer in getattr(store.bindings, "layers", ()):
            grouped_experts = getattr(layer, "grouped_experts", None)
            if grouped_experts is None:
                continue
            expert_ids = self._likely_hash_routed_expert_ids_for_token(
                stager,
                layer,
                token_id=token_id,
            )
            for expert_id in expert_ids:
                stager.pin_grouped_expert(layer.layer_index, expert_id)
            if expert_ids:
                stager.prefetch_grouped_experts(
                    grouped_experts,
                    DeepSeekV4FlashExpertPrefetchRequest(
                        layer_idx=layer.layer_index,
                        expert_ids=expert_ids,
                    ),
                )

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

    @staticmethod
    def _output_hidden_cuda(
        *,
        streams: torch.Tensor,
        output_hc_weight: torch.Tensor,
        output_hc_scale: torch.Tensor,
        output_hc_base: torch.Tensor,
        output_norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            streams,
            output_hc_weight,
            output_hc_scale,
            output_hc_base,
            output_norm_weight,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("DeepSeek V4 Flash output hidden inputs must be CUDA")
        streams_f32 = streams.to(torch.float32)
        flat = streams_f32.reshape(-1)
        flat = flat * torch.rsqrt(flat.pow(2).mean() + _RMS_NORM_EPS)
        pre = output_hc_weight.to(torch.float32).matmul(flat)
        weights = (
            torch.sigmoid(
                pre * output_hc_scale.to(torch.float32)[0]
                + output_hc_base.to(torch.float32)
            )
            + 1e-6
        )
        hidden = (weights.reshape(4, 1) * streams_f32).sum(dim=0)
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean() + _RMS_NORM_EPS)
        return hidden * output_norm_weight.to(torch.float32)

    @staticmethod
    def _raw_q8_chunk(
        raw_payload: torch.Tensor,
        *,
        row_start: int,
        row_end: int,
        row_bytes: int,
    ) -> torch.Tensor:
        byte_start = row_start * row_bytes
        byte_end = row_end * row_bytes
        if byte_start < 0 or byte_end <= byte_start or byte_end > raw_payload.numel():
            raise RuntimeError(
                "DeepSeek V4 Flash raw output chunk range is invalid: "
                f"[{byte_start}, {byte_end}) for payload {raw_payload.numel()}"
            )
        return raw_payload[byte_start:byte_end]

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
        stage_raw_payload = getattr(stager, "stage_q8_raw_payload", None)
        if callable(stage_raw_payload):
            output_hidden = self._output_hidden_cuda(
                streams=streams,
                output_hc_weight=output_hc_weight,
                output_hc_scale=output_hc_scale,
                output_hc_base=output_hc_base,
                output_norm_weight=output_norm_weight,
            )
            raw_payload = stage_raw_payload(tensor)
            logits_chunks: list[torch.Tensor] = []
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                chunk_rows = row_end - row_start
                chunk_raw = self._raw_q8_chunk(
                    raw_payload,
                    row_start=row_start,
                    row_end=row_end,
                    row_bytes=row_bytes,
                )
                chunk_logits = q8_0_raw_linear(
                    chunk_raw,
                    output_hidden,
                    rows=chunk_rows,
                    columns=columns,
                    block_size=_Q8_0_BLOCK_SIZE,
                )
                if not chunk_logits.is_cuda:
                    raise RuntimeError(
                        "DeepSeek V4 Flash raw output chunk returned CPU logits"
                    )
                logits_chunks.append(chunk_logits.reshape(-1))
            if not logits_chunks:
                raise RuntimeError("DeepSeek V4 Flash GPU output produced no chunks")
            return torch.cat(logits_chunks, dim=0).reshape(1, -1)

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
        stage_raw_payload = getattr(stager, "stage_q8_raw_payload", None)
        if callable(stage_raw_payload):
            output_hidden = self._output_hidden_cuda(
                streams=streams,
                output_hc_weight=output_hc_weight,
                output_hc_scale=output_hc_scale,
                output_hc_base=output_hc_base,
                output_norm_weight=output_norm_weight,
            )
            raw_payload = stage_raw_payload(tensor)
            best_value: torch.Tensor | None = None
            best_token: torch.Tensor | None = None
            for row_start in range(0, rows, _OUTPUT_PROJECTION_CHUNK_ROWS):
                row_end = min(row_start + _OUTPUT_PROJECTION_CHUNK_ROWS, rows)
                chunk_rows = row_end - row_start
                chunk_raw = self._raw_q8_chunk(
                    raw_payload,
                    row_start=row_start,
                    row_end=row_end,
                    row_bytes=row_bytes,
                )
                chunk_logits = q8_0_raw_linear(
                    chunk_raw,
                    output_hidden,
                    rows=chunk_rows,
                    columns=columns,
                    block_size=_Q8_0_BLOCK_SIZE,
                ).reshape(-1)
                if not chunk_logits.is_cuda:
                    raise RuntimeError(
                        "DeepSeek V4 Flash raw output chunk returned CPU logits"
                    )
                chunk_value, chunk_index = torch.max(chunk_logits, dim=0)
                chunk_token = chunk_index.to(torch.long) + row_start
                if best_value is None or best_token is None:
                    best_value = chunk_value
                    best_token = chunk_token
                else:
                    take_chunk = chunk_value > best_value
                    best_value = torch.where(take_chunk, chunk_value, best_value)
                    best_token = torch.where(take_chunk, chunk_token, best_token)
            if best_token is None:
                raise RuntimeError("DeepSeek V4 Flash GPU output produced no chunks")
            return best_token.to(device=device, dtype=torch.long).reshape(())

        payload: memoryview | None = None
        best_value: torch.Tensor | None = None
        best_token: torch.Tensor | None = None
        output_hidden: torch.Tensor | None = None
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
                chunk_value: torch.Tensor | None = None
                if output_hidden is None and hasattr(self.gpu_backend, "output_hidden"):
                    output_hidden = self.gpu_backend.output_hidden(
                        streams=streams,
                        lm_head_values=lm_head_values,
                        lm_head_scales=lm_head_scales,
                        output_hc_weight=output_hc_weight,
                        output_hc_scale=output_hc_scale,
                        output_hc_base=output_hc_base,
                        output_norm_weight=output_norm_weight,
                        block_size=_Q8_0_BLOCK_SIZE,
                    )
                    if not output_hidden.is_cuda:
                        raise RuntimeError(
                            "DeepSeek V4 Flash GPU output hidden returned CPU tensor"
                        )

                if output_hidden is not None and hasattr(
                    self.gpu_backend, "output_argmax_from_hidden"
                ):
                    token, value = self.gpu_backend.output_argmax_from_hidden(
                        hidden=output_hidden,
                        lm_head_values=lm_head_values,
                        lm_head_scales=lm_head_scales,
                        block_size=_Q8_0_BLOCK_SIZE,
                        row_offset=row_start,
                    )
                    if not token.is_cuda or not value.is_cuda:
                        raise RuntimeError(
                            "DeepSeek V4 Flash GPU output chunk returned CPU argmax"
                        )
                    chunk_token = token.to(device=device, dtype=torch.long).reshape(())
                    chunk_value = value.to(device=device, dtype=torch.float32).reshape(
                        ()
                    )
                elif hasattr(self.gpu_backend, "output_argmax_with_value"):
                    token, value = self.gpu_backend.output_argmax_with_value(
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
                    if not token.is_cuda or not value.is_cuda:
                        raise RuntimeError(
                            "DeepSeek V4 Flash GPU output chunk returned CPU argmax"
                        )
                    chunk_token = token.to(device=device, dtype=torch.long).reshape(())
                    chunk_value = value.to(device=device, dtype=torch.float32).reshape(
                        ()
                    )
                elif hasattr(self.gpu_backend, "output_argmax"):
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

                if chunk_value is None:
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
                if chunk_value is None or chunk_token is None:
                    raise RuntimeError("DeepSeek V4 Flash output argmax failed")
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
        pre = store.decode_matrix(hc.fn).transpose(0, 1).to(torch.float32).matmul(flat)
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
        scales = (
            scale_bytes.view(torch.float16)
            .to(torch.float32)
            .reshape(
                rows,
                blocks_per_row,
            )
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
