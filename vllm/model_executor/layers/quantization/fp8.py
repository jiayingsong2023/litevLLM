# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_fi_trtllm_fp8_per_tensor_moe,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)

class Fp8Config(QuantizationConfig):
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    Supports loading kv-cache scaling factors from FP8 checkpoints.
