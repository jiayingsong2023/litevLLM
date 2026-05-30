# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.lite_linear import LiteLinear

from .config import Gemma4LayerConfig
from vllm.model_executor.models.lite_config import LiteConfig
from .policy_utils import _gemma4_kernel_policy_truthy, _gemma4_model_policy_truthy


class Gemma4MLP(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.intermediate_size = int(config.intermediate_size)
        self.gate_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.gate_proj",
        )
        self.up_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.up_proj",
        )
        self.down_proj = LiteLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.down_proj",
        )
        self._layer_config = Gemma4LayerConfig()

    def _apply_activation(self, gate: torch.Tensor) -> torch.Tensor:
        # Keep activation dispatch isolated so the fused and unfused branches
        # share one implementation.
        if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            return F.gelu(gate, approximate="tanh")
        return F.silu(gate)

    def forward(
        self,
        x: torch.Tensor,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> torch.Tensor:
        # Step 4 pair fusion: concat gate_proj and up_proj into a single
        # quantized GEMM, halving the per-layer kernel launches for AWQ/int4
        # weights.
        #
        # LoRA-activity detection is delegated to the helper because the
        # input_batch_builder always hands us a list like ``[None, None]``
        # for non-LoRA requests (see vllm/engine/input_batch_builder.py).
        # The outer env gate only toggles the optimization wholesale.
        #
        # The helper returns None when structural guards trip (mismatched
        # shapes, high-fidelity flag, active LoRA, etc.) and the caller
        # falls back to the two-matmul path below.
        if _gemma4_kernel_policy_truthy(inf_config, "awq_fused_gate_up", default=True):
            from vllm.model_executor.models._fused_awq_pair import (
                try_fused_awq_gate_up_activation,
            )

            h = try_fused_awq_gate_up_activation(
                x,
                self.gate_proj,
                self.up_proj,
                activation=self.hidden_act,
                lora_mapping=lora_mapping,
                inf_config=inf_config,
            )
            if h is not None:
                return self.down_proj(h, lora_mapping, inf_config=inf_config)

        if _gemma4_model_policy_truthy(inf_config, "mlp_pair_fusion", default=True):
            from vllm.model_executor.models._fused_awq_pair import (
                try_fused_awq_pair_matmul,
            )

            gu = try_fused_awq_pair_matmul(
                x,
                self.gate_proj,
                self.up_proj,
                self,
                "mlp_gate_up",
                lora_mapping=lora_mapping,
                inf_config=inf_config,
            )
            if gu is not None:
                gate, up = torch.split(gu, self.intermediate_size, dim=-1)
                act = self._apply_activation(gate)
                return self.down_proj(act * up, lora_mapping, inf_config=inf_config)

        gate = self.gate_proj(x, lora_mapping, inf_config=inf_config)
        up = self.up_proj(x, lora_mapping, inf_config=inf_config)
        act = self._apply_activation(gate)
        return self.down_proj(act * up, lora_mapping, inf_config=inf_config)
