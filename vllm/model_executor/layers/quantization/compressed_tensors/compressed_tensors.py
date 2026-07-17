# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.tensor import AWQWeight, PackedInt4Weight


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _read_bits_group_size(cfg: dict[str, Any]) -> tuple[int, int]:
    weight_bits = _as_int(cfg.get("weight_bits", cfg.get("bits", 4)), 4)
    group_size = _as_int(cfg.get("group_size", 128), 128)
    groups = cfg.get("config_groups")
    if isinstance(groups, dict):
        for g in groups.values():
            if not isinstance(g, dict):
                continue
            w = g.get("weights")
            if isinstance(w, dict):
                if w.get("num_bits") is not None:
                    weight_bits = _as_int(w.get("num_bits"), weight_bits)
                if w.get("group_size") is not None:
                    group_size = _as_int(w.get("group_size"), group_size)
                break
    return weight_bits, group_size


def _compressed_tensors_high_fidelity_enabled(config: object | None = None) -> bool:
    # Throughput-first default for pack-quantized checkpoints:
    # keep fused int4 path enabled unless policy explicitly requests strict mode.
    policy = getattr(config, "kernel_policy", None)
    if isinstance(config, dict):
        policy = config.get("kernel_policy", config)
    if isinstance(policy, dict) and "compressed_tensors_high_fidelity" in policy:
        raw = policy["compressed_tensors_high_fidelity"]
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return False


class CompressedTensorsConfig(QuantizationConfig):
    def __init__(self, weight_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.weight_bits = int(weight_bits)
        self.group_size = int(group_size)
        self.pack_factor = 32 // self.weight_bits

    def get_name(self) -> str:
        return "compressed-tensors"

    def init_layer(self, layer: nn.Module):
        layer.qweight = nn.Parameter(
            torch.zeros((1, 1), dtype=torch.int32), requires_grad=False
        )
        layer.scales = nn.Parameter(
            torch.zeros((1, 1), dtype=torch.float16), requires_grad=False
        )
        layer.qzeros = None
        layer.weight_shape = None
        layer.group_size = self.group_size
        layer._quant_weight = None

    def _build_quant_weight(self, layer: nn.Module):
        qweight = getattr(layer, "qweight", None)
        scales = getattr(layer, "scales", None)
        qzeros = getattr(layer, "qzeros", None)
        weight_shape = getattr(layer, "weight_shape", None)
        if qweight is None or scales is None:
            raise RuntimeError(
                f"CompressedTensors weights are not initialized for layer "
                f"'{getattr(layer, 'prefix', '<unknown>')}'"
            )
        gs = int(getattr(layer, "group_size", self.group_size))
        high_fidelity = _compressed_tensors_high_fidelity_enabled(
            getattr(layer, "_fastinference_config", None)
        )
        if (
            qzeros is not None
            and isinstance(qzeros, torch.Tensor)
            and qzeros.numel() > 1
        ):
            return AWQWeight(
                qweight,
                scales,
                qzeros,
                gs,
                prefix=getattr(layer, "prefix", ""),
                high_fidelity=high_fidelity,
                profile_hint=str(getattr(layer, "awq_profile_hint", "")),
            )
        return PackedInt4Weight(
            qweight,
            scales,
            gs,
            original_shape=weight_shape,
            prefix=getattr(layer, "prefix", ""),
            high_fidelity=high_fidelity,
            profile_hint=str(getattr(layer, "awq_profile_hint", "")),
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args
        weight = getattr(layer, "_quant_weight", None)
        if weight is None:
            weight = self._build_quant_weight(layer)
            layer._quant_weight = weight
        runtime_config = kwargs.get("inf_config", kwargs.get("config"))
        if runtime_config is not None:
            layer._fastinference_config = runtime_config
        return weight.matmul(x, getattr(layer, "bias", None), config=runtime_config)

    def load_weights(
        self,
        layer: nn.Module,
        weights_iter: Iterable[tuple[str, torch.Tensor]],
        expert_idx: int | None = None,
        part: str | None = None,
    ):
        del expert_idx, part
        for name, loaded_weight in weights_iter:
            key = name.split(".")[-1]
            if key in ("weight_packed", "qweight"):
                layer.qweight = nn.Parameter(
                    loaded_weight.contiguous().to(torch.int32), requires_grad=False
                )
            elif key in ("weight_scale", "scales"):
                layer.scales = nn.Parameter(
                    loaded_weight.contiguous(), requires_grad=False
                )
            elif key in ("weight_zero", "qzeros", "zeros"):
                layer.qzeros = nn.Parameter(
                    loaded_weight.contiguous().to(torch.int32), requires_grad=False
                )
            elif key == "weight_shape":
                vals = loaded_weight.view(-1).tolist()
                if len(vals) >= 2:
                    layer.weight_shape = (int(vals[0]), int(vals[1]))
            elif key == "bias":
                layer.bias = nn.Parameter(
                    loaded_weight.contiguous(), requires_grad=False
                )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CompressedTensorsConfig:
        bits, group_size = _read_bits_group_size(config)
        return cls(weight_bits=bits, group_size=group_size)
