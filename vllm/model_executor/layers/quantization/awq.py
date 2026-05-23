# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.tensor import (
    AWQWeight,
    PackedInt4Weight,
)

class AWQConfig(QuantizationConfig):
    def __init__(self, weight_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // weight_bits

    def get_name(self) -> str: return "awq"

    def init_layer(self, layer: nn.Module):
        # Register as parameters with 2D dummy size so they have valid strides
        # AWQ weights and zeros MUST be integers for bitwise shifts
        layer.qweight = nn.Parameter(torch.zeros((1, 1), dtype=torch.int32), requires_grad=False)
        layer.scales = nn.Parameter(torch.zeros((1, 1), dtype=torch.float16), requires_grad=False)
        layer.qzeros = None
        layer.weight_shape = None
        # Propagate default group size so that layout checks and kernels can use it.
        if not hasattr(layer, "group_size"):
            layer.group_size = self.group_size
        layer._quant_weight = None

    def apply(
        self, layer: nn.Module, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        del args
        # Check if already built
        weight = getattr(layer, "_quant_weight", None)
        if weight is None:
            qweight = getattr(layer, "qweight", None)
            if qweight is None or qweight.numel() <= 1:
                # Fallback to standard if possible
                if hasattr(layer, "weight") and layer.weight is not None and layer.weight.numel() > 1:
                    return torch.nn.functional.linear(x, layer.weight, layer.bias)
                raise RuntimeError(f"AWQ weight not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'")

            # Determine effective group size for this layer
            effective_group_size = getattr(layer, "group_size", self.group_size)

            qzeros = getattr(layer, "qzeros", None)

            # Optional sanity check: ensure packed columns are divisible by group size.
            try:
                n_rows, n_cols_packed = qweight.shape
                n_cols = n_cols_packed * 8
                if n_cols % effective_group_size != 0:
                    raise RuntimeError(
                        f"AWQ group_size={effective_group_size} does not divide "
                        f"dequantized columns ({n_cols}) for layer '{getattr(layer, 'prefix', '<unknown>')}'"
                    )
            except Exception as e:
                raise RuntimeError(f"AWQ layout check failed for layer '{getattr(layer, 'prefix', '<unknown>')}': {e}")

            if qzeros is not None and qzeros.numel() > 1:
                weight = AWQWeight(
                    qweight,
                    getattr(layer, "scales"),
                    qzeros,
                    effective_group_size,
                    prefix=getattr(layer, "prefix", ""),
                    high_fidelity=bool(getattr(layer, "force_high_fidelity_awq", False)),
                    profile_hint=str(getattr(layer, "awq_profile_hint", "")),
                )
            else:
                weight = PackedInt4Weight(
                    qweight,
                    getattr(layer, "scales"),
                    effective_group_size,
                    original_shape=getattr(layer, "weight_shape", None),
                    prefix=getattr(layer, "prefix", ""),
                    high_fidelity=bool(getattr(layer, "force_high_fidelity_awq", False)),
                    profile_hint=str(getattr(layer, "awq_profile_hint", "")),
                )
            layer._quant_weight = weight

        runtime_config = kwargs.get("inf_config", kwargs.get("config"))
        return weight.matmul(x, layer.bias, config=runtime_config)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in weights_iter:
            if "qweight" in name: layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "scales" in name: layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "qzeros" in name: layer.qzeros = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "weight_shape" in name: layer.weight_shape = tuple(int(x) for x in loaded_weight.view(-1).tolist())
            elif "bias" in name: layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        # HF / compressed-tensors often nest group_size under config_groups.*.weights
        weight_bits = int(config.get("weight_bits", config.get("bits", 4)))
        group_size = int(config.get("group_size", 128))
        groups = config.get("config_groups")
        if isinstance(groups, dict):
            for g in groups.values():
                if not isinstance(g, dict):
                    continue
                w = g.get("weights")
                if isinstance(w, dict):
                    if w.get("group_size") is not None:
                        group_size = int(w["group_size"])
                    if w.get("num_bits") is not None:
                        weight_bits = int(w["num_bits"])
                    break
        return cls(weight_bits=weight_bits, group_size=group_size)
