# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .gpu_runtime import DeepSeekV4FlashGPURequestState
from .model import DeepSeekV4FlashForCausalLM


class DeepSeekV4FlashDecodeGraph:
    """CUDA/HIP graph capture for one steady-state decode step.

    The graph assumes:
    - input token is a CUDA scalar long tensor (placeholder)
    - RoPE tables are precomputed and read from state (not captured as constants)
    - KV window tensors are stable input objects copied before replay
    - expert payload tensors are stable cached objects whose bytes are copied
      before replay
    """

    def __init__(
        self,
        *,
        device: torch.device,
    ) -> None:
        self.token_id_placeholder = torch.empty(
            (),
            dtype=torch.long,
            device=device,
        )
        self.output_token = torch.empty(
            (),
            dtype=torch.long,
            device=device,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.kv_rows_by_layer: dict[int, torch.Tensor] = {}
        self.extra_kv_rows_by_layer: dict[int, torch.Tensor | None] = {}

    @classmethod
    def capture(
        cls,
        model: DeepSeekV4FlashForCausalLM,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
        kv_rows_by_layer: dict[int, torch.Tensor | None] | None = None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] | None = None,
    ) -> DeepSeekV4FlashDecodeGraph:
        """Capture a single decode step as a CUDA/HIP graph.

        The capture runs one warm-up step with ``advance_state=False`` so the
        request state stays synchronized to ``token_idx``. The caller must
        advance ``state`` after capture if it intends to continue decoding.
        """
        instance = cls(device=device)
        token_id_tensor = instance.token_id_placeholder
        # Initialize the placeholder with a valid token id so that async range
        # assertions in the model step succeed during warm-up and capture.
        token_id_tensor.fill_(0)

        # Allocate stable KV window tensors from the initial windows so the
        # graph records operations on stable objects.
        instance._prepare_kv_windows(
            kv_rows_by_layer=kv_rows_by_layer,
            extra_kv_rows_by_layer=extra_kv_rows_by_layer,
        )
        # Stage the current token's expert payloads and copy the current KV
        # windows into the stable tensors.
        instance.prepare_replay(
            model=model,
            state=state,
            token_idx=token_idx,
            token_id_tensor=token_id_tensor,
            device=device,
        )

        def _capture_step() -> None:
            out = model._forward_kernel_token_step_token_tensor(
                token_id_tensor=token_id_tensor,
                state=state,
                token_idx=token_idx,
                device=device,
                advance_state=False,
                kv_rows_by_layer=instance.kv_rows_by_layer,
                extra_kv_rows_by_layer=instance.extra_kv_rows_by_layer,
            )
            instance.output_token.copy_(out.reshape(()))

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # Warm-up: identical to the captured function but not recorded.
            _capture_step()
        torch.cuda.current_stream().wait_stream(capture_stream)

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            instance.graph.capture_begin()
            _capture_step()
            instance.graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        return instance

    def _prepare_kv_windows(
        self,
        *,
        kv_rows_by_layer: dict[int, torch.Tensor | None] | None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor | None] | None,
    ) -> None:
        """Allocate stable KV window tensors from the initial windows."""
        if kv_rows_by_layer is not None:
            for layer_idx, tensor in kv_rows_by_layer.items():
                if tensor is None:
                    continue
                stable = torch.empty_like(tensor)
                stable.copy_(tensor)
                self.kv_rows_by_layer[layer_idx] = stable
        if extra_kv_rows_by_layer is not None:
            for layer_idx, tensor in extra_kv_rows_by_layer.items():
                if tensor is None:
                    self.extra_kv_rows_by_layer[layer_idx] = None
                else:
                    stable = torch.empty_like(tensor)
                    stable.copy_(tensor)
                    self.extra_kv_rows_by_layer[layer_idx] = stable

    def prepare_replay(
        self,
        model: DeepSeekV4FlashForCausalLM,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        token_id_tensor: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Update stable inputs before ``graph.replay()``.

        This stages the current token's expert payloads and copies the current
        KV windows into the stable tensors that the graph was captured with.
        """
        model._stage_graph_moe_payloads(
            state=state,
            token_id=int(token_id_tensor.item()),
            device=device,
        )
        kv_rows_by_layer, extra_kv_rows_by_layer = model._materialize_decode_kv_windows(
            state=state,
            token_idx=token_idx,
        )
        for layer_idx, current in kv_rows_by_layer.items():
            stable = self.kv_rows_by_layer.get(layer_idx)
            if stable is not None and stable.shape == current.shape:
                stable.copy_(current)
        for layer_idx, current in extra_kv_rows_by_layer.items():
            if current is None:
                continue
            stable = self.extra_kv_rows_by_layer.get(layer_idx)
            if stable is not None and stable.shape == current.shape:
                stable.copy_(current)

    def replay(self, token_id_tensor: torch.Tensor) -> torch.Tensor:
        self.token_id_placeholder.copy_(token_id_tensor)
        self.graph.replay()
        return self.output_token.clone()
