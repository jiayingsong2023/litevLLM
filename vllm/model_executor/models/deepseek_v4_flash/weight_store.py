# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mmap
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import torch

from .config import DEEPSEEK_V4_FLASH_SHAPE
from .gguf_reader import (
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashGGUF,
    DeepSeekV4FlashTensor,
    ggml_tensor_nbytes,
    read_deepseek_v4_flash_gguf_from_view,
)
from .quant import (
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
    q8_0_dequantize_reference,
    q8_0_matrix_from_gguf_payload,
)


class DeepSeekV4FlashWeightStoreError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekV4FlashExpertTensors:
    expert_id: int
    gate: DeepSeekV4FlashTensor
    down: DeepSeekV4FlashTensor
    up: DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashGroupedExpertTensors:
    gate: DeepSeekV4FlashTensor
    down: DeepSeekV4FlashTensor
    up: DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashHyperConnectionTensors:
    fn: DeepSeekV4FlashTensor
    base: DeepSeekV4FlashTensor
    scale: DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashCompressorTensors:
    ape: DeepSeekV4FlashTensor
    gate: DeepSeekV4FlashTensor
    kv: DeepSeekV4FlashTensor
    norm: DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashIndexerTensors:
    query_b: DeepSeekV4FlashTensor
    projection: DeepSeekV4FlashTensor
    compressor: DeepSeekV4FlashCompressorTensors


@dataclass(frozen=True)
class DeepSeekV4FlashLayerSemanticBindings:
    layer_index: int
    attention_norm: DeepSeekV4FlashTensor | None = None
    attention_sinks: DeepSeekV4FlashTensor | None = None
    attention_query: DeepSeekV4FlashTensor | None = None
    attention_query_a: DeepSeekV4FlashTensor | None = None
    attention_query_a_norm: DeepSeekV4FlashTensor | None = None
    attention_query_b: DeepSeekV4FlashTensor | None = None
    attention_key_value: DeepSeekV4FlashTensor | None = None
    attention_key_value_a_norm: DeepSeekV4FlashTensor | None = None
    attention_output: DeepSeekV4FlashTensor | None = None
    attention_output_a: DeepSeekV4FlashTensor | None = None
    attention_output_b: DeepSeekV4FlashTensor | None = None
    attention_compressor: DeepSeekV4FlashCompressorTensors | None = None
    indexer: DeepSeekV4FlashIndexerTensors | None = None
    ffn_norm: DeepSeekV4FlashTensor | None = None
    router: DeepSeekV4FlashTensor | None = None
    routed_experts: dict[int, DeepSeekV4FlashExpertTensors] | None = None
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors | None = None
    shared_experts: DeepSeekV4FlashGroupedExpertTensors | None = None
    expert_token_to_expert_ids: DeepSeekV4FlashTensor | None = None
    expert_probs_bias: DeepSeekV4FlashTensor | None = None
    attention_hyper_connection: DeepSeekV4FlashHyperConnectionTensors | None = None
    ffn_hyper_connection: DeepSeekV4FlashHyperConnectionTensors | None = None


@dataclass(frozen=True)
class DeepSeekV4FlashSemanticBindings:
    token_embedding: DeepSeekV4FlashTensor
    representative_layer_tensor: DeepSeekV4FlashTensor
    attention_query_by_layer: dict[int, DeepSeekV4FlashTensor]
    layers: tuple[DeepSeekV4FlashLayerSemanticBindings, ...]
    output_norm: DeepSeekV4FlashTensor | None = None
    output_head: DeepSeekV4FlashTensor | None = None


@dataclass(frozen=True)
class DeepSeekV4FlashExpertCachePolicy:
    max_dynamic_bytes: int
    pinned_experts: tuple[tuple[int, int], ...] = ()
    defer_eviction_during_forward: bool = True

    def __post_init__(self) -> None:
        if self.max_dynamic_bytes < 0:
            raise ValueError("expert cache bytes must be non-negative")
        seen: set[tuple[int, int]] = set()
        for layer_idx, expert_id in self.pinned_experts:
            if layer_idx < 0 or layer_idx >= DEEPSEEK_V4_FLASH_SHAPE.num_layers:
                raise ValueError(f"layer index out of range: {layer_idx}")
            if expert_id < 0 or expert_id >= DEEPSEEK_V4_FLASH_SHAPE.num_experts:
                raise ValueError(f"expert id out of range: {expert_id}")
            key = (layer_idx, expert_id)
            if key in seen:
                raise ValueError(f"duplicate pinned experts are not allowed: {key}")
            seen.add(key)


@dataclass(frozen=True)
class DeepSeekV4FlashWeightStoreDiagnostics:
    tensor_count: int
    file_size_bytes: int
    mmap_size_bytes: int
    bound_tensor_count: int
    missing_required_semantic_tensors: tuple[str, ...]
    tensor_type_counts: dict[int, int]
    tensor_type_samples: dict[int, tuple[DeepSeekV4FlashTensor, ...]]
    unaligned_tensor_offsets: tuple[str, ...]


_REQUIRED_TOKEN_TENSOR = "token_embd.weight"
_REQUIRED_LAYER_0_ATTENTION_QUERY_CANDIDATES = (
    "blk.0.attn_q.weight",
    "blk.0.attn_q_a.weight",
)
_REQUIRED_TENSORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("token_embedding", (_REQUIRED_TOKEN_TENSOR,)),
    (
        "layer_0_attention_query",
        _REQUIRED_LAYER_0_ATTENTION_QUERY_CANDIDATES,
    ),
)


def grouped_expert_payload_offset(
    *,
    expert_id: int,
    projection_dims: tuple[int, int, int],
    projection_type: int,
) -> int:
    if len(projection_dims) != 3:
        raise ValueError(
            f"projection_dims must have exactly 3 dimensions; got {projection_dims}"
        )
    input_size, output_size, expert_count = projection_dims
    if expert_id < 0 or expert_id >= expert_count:
        raise ValueError(
            f"expert id out of range: {expert_id}; expected [0, {expert_count})"
        )
    expert_nbytes = ggml_tensor_nbytes(
        (input_size, output_size),
        projection_type,
    )
    return expert_id * expert_nbytes


class DeepSeekV4FlashWeightStore:
    def __init__(
        self,
        *,
        model: DeepSeekV4FlashGGUF,
        file: BinaryIO,
        mmap_obj: mmap.mmap,
        bindings: DeepSeekV4FlashSemanticBindings,
        diagnostics: DeepSeekV4FlashWeightStoreDiagnostics,
    ) -> None:
        self.model = model
        self._file = file
        self._mmap = mmap_obj
        self.bindings = bindings
        self.diagnostics = diagnostics

    @property
    def mmap_size_bytes(self) -> int:
        return self._mmap.size()

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        """Return a view over tensor bytes.

        The caller owns the returned memoryview and must release it before
        closing the store.
        """
        start = self.model.data_offset + tensor.offset
        end = start + tensor.nbytes
        mmap_size = self._mmap.size()
        if start < 0 or end < start or end > mmap_size:
            raise DeepSeekV4FlashWeightStoreError(
                f"tensor {tensor.name} payload range [{start}, {end}) "
                f"from data offset {self.model.data_offset} exceeds mmap size "
                f"{mmap_size}"
            )
        view = memoryview(self._mmap)
        try:
            return view[start:end]
        finally:
            view.release()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        payload = self.tensor_payload(tensor)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable.*",
                    category=UserWarning,
                )
                data = torch.frombuffer(payload, dtype=dtype).clone()
        finally:
            payload.release()
        return data.reshape(shape or tensor.dims)

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        if len(tensor.dims) != 3:
            raise DeepSeekV4FlashWeightStoreError(
                f"grouped expert tensor {tensor.name} must have dims "
                f"(input, output, expert_count); got {tensor.dims}"
            )
        input_size, output_size, _expert_count = tensor.dims
        expert_offset = grouped_expert_payload_offset(
            expert_id=expert_id,
            projection_dims=tensor.dims,
            projection_type=tensor.tensor_type,
        )
        expert_nbytes = ggml_tensor_nbytes(
            (input_size, output_size),
            tensor.tensor_type,
        )
        payload = self.tensor_payload(tensor)
        expert_payload: memoryview | None = None
        try:
            expert_payload = payload[expert_offset : expert_offset + expert_nbytes]
            expert_bytes = expert_payload.tobytes()
        finally:
            if expert_payload is not None:
                expert_payload.release()
            payload.release()

        if tensor.tensor_type == GGML_TYPE_IQ2_XXS:
            return iq2_xxs_matrix_from_gguf_payload(
                expert_bytes,
                rows=output_size,
                columns=input_size,
            )
        if tensor.tensor_type == GGML_TYPE_Q2_K:
            return q2_k_matrix_from_gguf_payload(
                expert_bytes,
                rows=output_size,
                columns=input_size,
            )
        if tensor.tensor_type == GGML_TYPE_Q8_0:
            values, scales = q8_0_matrix_from_gguf_payload(
                expert_bytes,
                rows=output_size,
                columns=input_size,
            )
            return q8_0_dequantize_reference(values, scales)
        raise DeepSeekV4FlashWeightStoreError(
            f"unsupported grouped expert tensor type for {tensor.name}: "
            f"{tensor.tensor_type}"
        )

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            self._file.close()

    def __enter__(self) -> DeepSeekV4FlashWeightStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _bind_required_tensors(
    model: DeepSeekV4FlashGGUF,
) -> tuple[DeepSeekV4FlashSemanticBindings | None, tuple[str, ...]]:
    missing = tuple(
        semantic_name
        for semantic_name, tensor_names in _REQUIRED_TENSORS
        if all(tensor_name not in model.tensors for tensor_name in tensor_names)
    )
    if missing:
        return None, missing

    layers = _bind_layers(model)
    attention_query_by_layer = _bind_attention_query_tensors(layers)
    representative_layer_tensor = _required_layer_0_attention_query(model)
    return (
        DeepSeekV4FlashSemanticBindings(
            token_embedding=model.tensors[_REQUIRED_TOKEN_TENSOR],
            representative_layer_tensor=representative_layer_tensor,
            attention_query_by_layer=attention_query_by_layer,
            layers=layers,
            output_norm=model.tensors.get("output_norm.weight"),
            output_head=model.tensors.get("output.weight"),
        ),
        (),
    )


def _required_layer_0_attention_query(
    model: DeepSeekV4FlashGGUF,
) -> DeepSeekV4FlashTensor:
    for tensor_name in _REQUIRED_LAYER_0_ATTENTION_QUERY_CANDIDATES:
        tensor = model.tensors.get(tensor_name)
        if tensor is not None:
            return tensor
    raise AssertionError("layer 0 attention query was not validated")


def _tensor(
    model: DeepSeekV4FlashGGUF,
    layer_idx: int,
    suffix: str,
) -> DeepSeekV4FlashTensor | None:
    return model.tensors.get(f"blk.{layer_idx}.{suffix}")


def _bind_grouped_experts(
    model: DeepSeekV4FlashGGUF,
    layer_idx: int,
    *,
    gate_suffix: str,
    down_suffix: str,
    up_suffix: str,
) -> DeepSeekV4FlashGroupedExpertTensors | None:
    gate = _tensor(model, layer_idx, gate_suffix)
    down = _tensor(model, layer_idx, down_suffix)
    up = _tensor(model, layer_idx, up_suffix)
    if gate is None or down is None or up is None:
        return None
    return DeepSeekV4FlashGroupedExpertTensors(gate=gate, down=down, up=up)


def _bind_hyper_connection(
    model: DeepSeekV4FlashGGUF,
    layer_idx: int,
    *,
    prefix: str,
) -> DeepSeekV4FlashHyperConnectionTensors | None:
    fn = _tensor(model, layer_idx, f"{prefix}_fn.weight")
    base = _tensor(model, layer_idx, f"{prefix}_base.weight")
    scale = _tensor(model, layer_idx, f"{prefix}_scale.weight")
    if fn is None or base is None or scale is None:
        return None
    return DeepSeekV4FlashHyperConnectionTensors(fn=fn, base=base, scale=scale)


def _bind_compressor(
    model: DeepSeekV4FlashGGUF,
    layer_idx: int,
    *,
    prefix: str,
) -> DeepSeekV4FlashCompressorTensors | None:
    ape = _tensor(model, layer_idx, f"{prefix}_ape.weight")
    gate = _tensor(model, layer_idx, f"{prefix}_gate.weight")
    kv = _tensor(model, layer_idx, f"{prefix}_kv.weight")
    norm = _tensor(model, layer_idx, f"{prefix}_norm.weight")
    if ape is None or gate is None or kv is None or norm is None:
        return None
    return DeepSeekV4FlashCompressorTensors(
        ape=ape,
        gate=gate,
        kv=kv,
        norm=norm,
    )


def _bind_indexer(
    model: DeepSeekV4FlashGGUF,
    layer_idx: int,
) -> DeepSeekV4FlashIndexerTensors | None:
    query_b = _tensor(model, layer_idx, "indexer.attn_q_b.weight")
    projection = _tensor(model, layer_idx, "indexer.proj.weight")
    compressor = _bind_compressor(
        model,
        layer_idx,
        prefix="indexer_compressor",
    )
    if query_b is None or projection is None or compressor is None:
        return None
    return DeepSeekV4FlashIndexerTensors(
        query_b=query_b,
        projection=projection,
        compressor=compressor,
    )


def _bind_layers(
    model: DeepSeekV4FlashGGUF,
) -> tuple[DeepSeekV4FlashLayerSemanticBindings, ...]:
    layers: list[DeepSeekV4FlashLayerSemanticBindings] = []
    for layer_idx in range(model.shape.num_layers):
        attention_query = _tensor(model, layer_idx, "attn_q.weight")
        attention_query_a = _tensor(model, layer_idx, "attn_q_a.weight")
        attention_output = _tensor(model, layer_idx, "attn_output.weight")
        attention_output_a = _tensor(model, layer_idx, "attn_output_a.weight")
        layers.append(
            DeepSeekV4FlashLayerSemanticBindings(
                layer_index=layer_idx,
                attention_norm=_tensor(model, layer_idx, "attn_norm.weight"),
                attention_sinks=_tensor(model, layer_idx, "attn_sinks.weight"),
                attention_query=attention_query or attention_query_a,
                attention_query_a=attention_query_a,
                attention_query_a_norm=_tensor(
                    model,
                    layer_idx,
                    "attn_q_a_norm.weight",
                ),
                attention_query_b=_tensor(model, layer_idx, "attn_q_b.weight"),
                attention_key_value=_tensor(model, layer_idx, "attn_kv.weight"),
                attention_key_value_a_norm=_tensor(
                    model,
                    layer_idx,
                    "attn_kv_a_norm.weight",
                ),
                attention_output=attention_output or attention_output_a,
                attention_output_a=attention_output_a,
                attention_output_b=_tensor(model, layer_idx, "attn_output_b.weight"),
                attention_compressor=_bind_compressor(
                    model,
                    layer_idx,
                    prefix="attn_compressor",
                ),
                indexer=_bind_indexer(model, layer_idx),
                ffn_norm=_tensor(model, layer_idx, "ffn_norm.weight"),
                router=_tensor(model, layer_idx, "ffn_gate_inp.weight"),
                routed_experts={},
                grouped_experts=_bind_grouped_experts(
                    model,
                    layer_idx,
                    gate_suffix="ffn_gate_exps.weight",
                    down_suffix="ffn_down_exps.weight",
                    up_suffix="ffn_up_exps.weight",
                ),
                shared_experts=_bind_grouped_experts(
                    model,
                    layer_idx,
                    gate_suffix="ffn_gate_shexp.weight",
                    down_suffix="ffn_down_shexp.weight",
                    up_suffix="ffn_up_shexp.weight",
                ),
                expert_token_to_expert_ids=_tensor(
                    model, layer_idx, "ffn_gate_tid2eid.weight"
                ),
                expert_probs_bias=_tensor(model, layer_idx, "exp_probs_b.bias"),
                attention_hyper_connection=_bind_hyper_connection(
                    model,
                    layer_idx,
                    prefix="hc_attn",
                ),
                ffn_hyper_connection=_bind_hyper_connection(
                    model,
                    layer_idx,
                    prefix="hc_ffn",
                ),
            )
        )
    return tuple(layers)


def _bind_attention_query_tensors(
    layers: tuple[DeepSeekV4FlashLayerSemanticBindings, ...],
) -> dict[int, DeepSeekV4FlashTensor]:
    bound: dict[int, DeepSeekV4FlashTensor] = {}
    for layer in layers:
        if layer.attention_query is not None:
            bound[layer.layer_index] = layer.attention_query
    return bound


def _bound_tensor_count(
    bindings: DeepSeekV4FlashSemanticBindings | None,
) -> int:
    if bindings is None:
        return 0
    bound_names = {
        bindings.token_embedding.name,
        bindings.representative_layer_tensor.name,
    }
    if bindings.output_norm is not None:
        bound_names.add(bindings.output_norm.name)
    if bindings.output_head is not None:
        bound_names.add(bindings.output_head.name)
    for tensor in bindings.attention_query_by_layer.values():
        bound_names.add(tensor.name)
    for layer in bindings.layers:
        layer_tensors = (
            layer.attention_norm,
            layer.attention_sinks,
            layer.attention_query,
            layer.attention_query_a,
            layer.attention_query_a_norm,
            layer.attention_query_b,
            layer.attention_key_value,
            layer.attention_key_value_a_norm,
            layer.attention_output,
            layer.attention_output_a,
            layer.attention_output_b,
            layer.ffn_norm,
            layer.router,
            layer.expert_token_to_expert_ids,
            layer.expert_probs_bias,
        )
        for tensor in layer_tensors:
            if tensor is not None:
                bound_names.add(tensor.name)
        for expert_group in (layer.grouped_experts, layer.shared_experts):
            if expert_group is None:
                continue
            bound_names.add(expert_group.gate.name)
            bound_names.add(expert_group.down.name)
            bound_names.add(expert_group.up.name)
        for compressor in (layer.attention_compressor,):
            if compressor is None:
                continue
            bound_names.add(compressor.ape.name)
            bound_names.add(compressor.gate.name)
            bound_names.add(compressor.kv.name)
            bound_names.add(compressor.norm.name)
        if layer.indexer is not None:
            bound_names.add(layer.indexer.query_b.name)
            bound_names.add(layer.indexer.projection.name)
            bound_names.add(layer.indexer.compressor.ape.name)
            bound_names.add(layer.indexer.compressor.gate.name)
            bound_names.add(layer.indexer.compressor.kv.name)
            bound_names.add(layer.indexer.compressor.norm.name)
        for hc in (layer.attention_hyper_connection, layer.ffn_hyper_connection):
            if hc is None:
                continue
            bound_names.add(hc.fn.name)
            bound_names.add(hc.base.name)
            bound_names.add(hc.scale.name)
        if layer.routed_experts is None:
            continue
        for expert in layer.routed_experts.values():
            bound_names.add(expert.gate.name)
            bound_names.add(expert.down.name)
            bound_names.add(expert.up.name)
    return len(bound_names)


def _tensor_type_counts(model: DeepSeekV4FlashGGUF) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tensor in model.tensors.values():
        counts[tensor.tensor_type] = counts.get(tensor.tensor_type, 0) + 1
    return counts


def _tensor_type_samples(
    model: DeepSeekV4FlashGGUF,
    *,
    samples_per_type: int = 3,
) -> dict[int, tuple[DeepSeekV4FlashTensor, ...]]:
    samples: dict[int, list[DeepSeekV4FlashTensor]] = {}
    for tensor in model.tensors.values():
        type_samples = samples.setdefault(tensor.tensor_type, [])
        if len(type_samples) < samples_per_type:
            type_samples.append(tensor)
    return {
        tensor_type: tuple(type_samples)
        for tensor_type, type_samples in samples.items()
    }


def _unaligned_tensor_offsets(
    model: DeepSeekV4FlashGGUF,
    *,
    alignment: int = 32,
) -> tuple[str, ...]:
    return tuple(
        tensor.name
        for tensor in model.tensors.values()
        if tensor.offset % alignment != 0
    )


def open_deepseek_v4_flash_weight_store(path: Path | str) -> DeepSeekV4FlashWeightStore:
    gguf_path = Path(path)
    file = gguf_path.open("rb")
    try:
        mmap_obj = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        file.close()
        raise
    view = memoryview(mmap_obj)
    try:
        try:
            model = read_deepseek_v4_flash_gguf_from_view(gguf_path, view)
        finally:
            view.release()
        bindings, missing = _bind_required_tensors(model)
        diagnostics = DeepSeekV4FlashWeightStoreDiagnostics(
            tensor_count=len(model.tensors),
            file_size_bytes=gguf_path.stat().st_size,
            mmap_size_bytes=mmap_obj.size(),
            bound_tensor_count=_bound_tensor_count(bindings),
            missing_required_semantic_tensors=missing,
            tensor_type_counts=_tensor_type_counts(model),
            tensor_type_samples=_tensor_type_samples(model),
            unaligned_tensor_offsets=_unaligned_tensor_offsets(model),
        )
        if bindings is None:
            missing_list = ", ".join(
                f"{semantic_name} ({' or '.join(tensor_names)})"
                for semantic_name, tensor_names in _REQUIRED_TENSORS
                if semantic_name in missing
            )
            raise DeepSeekV4FlashWeightStoreError(
                "missing required inspect-only tensors: " + missing_list
            )
        return DeepSeekV4FlashWeightStore(
            model=model,
            file=file,
            mmap_obj=mmap_obj,
            bindings=bindings,
            diagnostics=diagnostics,
        )
    except Exception:
        mmap_obj.close()
        file.close()
        raise
