# SPDX-License-Identifier: Apache-2.0
import re
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PolicyMode = Literal["auto", "aggressive", "stable"]


@dataclass(frozen=True)
class ExecutionPolicyProfile:
    mode: PolicyMode
    max_active_requests: int
    max_tokens_cap: int


AGGRESSIVE_PROFILE = ExecutionPolicyProfile(
    mode="aggressive",
    max_active_requests=32,
    max_tokens_cap=1024,
)
STABLE_PROFILE = ExecutionPolicyProfile(
    mode="stable",
    max_active_requests=1,
    max_tokens_cap=256,
)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_model_size_billion(model_config: Any) -> float:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        hidden_size = _safe_float(getattr(hf_config, "hidden_size", None))
        num_layers = _safe_float(getattr(hf_config, "num_hidden_layers", None))
        vocab_size = _safe_float(getattr(hf_config, "vocab_size", None))
        intermediate_size = _safe_float(getattr(hf_config, "intermediate_size", None))

        if (
            hidden_size is not None
            and num_layers is not None
            and vocab_size is not None
            and hidden_size > 0
            and num_layers > 0
        ):
            if intermediate_size is None or intermediate_size <= 0:
                intermediate_size = 4 * hidden_size

            # Dense baseline approximation.
            layer_params = (4 * hidden_size * hidden_size) + (
                2 * hidden_size * intermediate_size
            )
            total_params = num_layers * layer_params + (vocab_size * hidden_size)

            # If model exposes MoE attributes, include expert weights.
            num_experts = _safe_float(
                getattr(hf_config, "n_routed_experts", None)
                or getattr(hf_config, "num_local_experts", None)
            )
            moe_intermediate_size = _safe_float(
                getattr(hf_config, "moe_intermediate_size", None)
            )
            if (
                num_experts is not None
                and moe_intermediate_size is not None
                and num_experts > 0
                and moe_intermediate_size > 0
            ):
                total_params += (
                    num_layers * num_experts * (2 * hidden_size * moe_intermediate_size)
                )

            return max(total_params / 1e9, 0.1)

    model_name = str(getattr(model_config, "model", ""))
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if match:
        return float(match.group(1))

    return 7.0


def get_total_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / (1024**3)


def select_loadtime_policy(
    model_config: Any,
    quant_config: Any | None,
    policy_mode: PolicyMode,
    total_gpu_memory_gb: float | None = None,
) -> ExecutionPolicyProfile:
    if policy_mode == "aggressive":
        return AGGRESSIVE_PROFILE
    if policy_mode == "stable":
        return STABLE_PROFILE

    total_gb = (
        total_gpu_memory_gb if total_gpu_memory_gb is not None else get_total_gpu_memory_gb()
    )
    model_b = estimate_model_size_billion(model_config)

    # On larger VRAM, we allow proportionally larger models to stay aggressive.
    aggressive_limit_b = total_gb * 0.27
    if quant_config is not None:
        aggressive_limit_b *= 1.45

    selected = AGGRESSIVE_PROFILE if model_b <= aggressive_limit_b else STABLE_PROFILE
    logger.info(
        "Load-time policy selected: mode=%s, model_size=%.2fB, total_gpu=%.1fGB, "
        "aggressive_limit=%.2fB",
        selected.mode,
        model_b,
        total_gb,
        aggressive_limit_b,
    )
    return selected
