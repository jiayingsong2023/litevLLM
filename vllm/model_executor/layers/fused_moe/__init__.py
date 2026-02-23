# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import activation_without_mul
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE,
)
from vllm.triton_utils import HAS_TRITON

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoERouter",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEPermuteExpertsUnpermute",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalize",
    "RoutingMethodType",
    "SharedFusedMoE",
    "ZeroExpertFusedMoE",
    "activation_without_mul",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts,
        fused_experts,
        get_config_file_name,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopk,
    )

    __all__ += [
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "GroupedTopk",
        "TritonExperts",
    ]
else:
    # Some model classes directly use the custom ops. Add placeholders
    # to avoid import errors.
    def _raise_exception(method: str):
        raise NotImplementedError(f"{method} is not implemented as lack of triton.")

    fused_topk = lambda *args, **kwargs: _raise_exception("fused_topk")
    fused_experts = lambda *args, **kwargs: _raise_exception("fused_experts")