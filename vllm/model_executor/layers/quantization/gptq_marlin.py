# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of

logger = init_logger(__name__)

def get_moe_quant_method(
    config: "GPTQMarlinConfig",
    layer: torch.nn.Module,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    if isinstance(layer, FusedMoE):
        # False = skip module, None = no override, else = Positive match
        if (
            get_dynamic_override(  # noqa: E712
                cloned_config,  # noqa: E712
                layer_name=prefix,
            )
            == False
        ):  # noqa: E712
            return UnquantizedFusedMoEMethod(layer.moe_config)

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return moe_method_cls(cloned_config, layer.moe_config)
    return None

class GPTQMarlinConfig(QuantizationConfig):

    Args:
        quant_config: The GPTQ Marlin quantization config.

    def __init__(
        self,
        quant_config: GPTQMarlinConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits == 4:
            self.quant_type = scalar_types.uint4b8
        elif self.quant_config.quant_type.size_bits == 8:
            self.quant_type = scalar_types.uint8b128
        else:
            raise ValueError("GPTQMarlinMoEMethod only supports int4 and int8 now.")
        self.input_dtype = None
        self.use_marlin = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_type == scalar_types.uint4b8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        device = layer.w13_qweight.device
        layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_type == scalar_types.uint4b8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        if self.input_dtype == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(layer.w13_qweight, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_qweight, inplace=True)
            layer.w13_scales.data = layer.w13_scales.data * 512
            layer.w2_scales.data = layer.w2_scales.data * 512

        # Process act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            num_experts = layer.w13_g_idx.shape[0]
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            # Reset g_idx related tensors
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
        # Repack weights
        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        # The modular kernel expects w13_weight and w2_weight,
        # but GPTQ uses w13_qweight and w2_qweight
        # Alias for modular kernel
        layer.w13_weight = layer.w13_qweight
        # Alias for modular kernel
        layer.w2_weight = layer.w2_qweight

        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.input_dtype == torch.int8 and layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
            layer.register_parameter(
                "w13_input_global_scale",
                torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
            )

        replace_parameter(layer, "w13_scales", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.w2_scales.shape[1]
            * (
                self.quant_config.group_size
                if self.quant_config.group_size != -1
                else self.quant_config.pack_factor
            ),
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.input_dtype == torch.int8 and layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )
            layer.register_parameter(
                "w2_input_global_scale",
                torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
            )

        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
            layer.w13_bias.data = marlin_permute_bias(layer.w13_bias)

        if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
            layer.w2_bias.data = marlin_permute_bias(layer.w2_bias)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        from vllm.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: torch.nn.Module,
    ):
        # Only use modular kernels when LoRA is enabled
        # Without LoRA, GPTQ's own apply() method works fine and is more efficient
        if not self.moe.is_lora_enabled:
            raise NotImplementedError(
                "GPTQ-Marlin uses its own apply() method when LoRA is not enabled. "
                "Modular kernels are only used for LoRA support."
            )

        # The modular marlin kernels do not support 8-bit weights.
        if self.quant_config.weight_bits == 8:
            raise NotImplementedError(
                "GPTQ-Marlin kernel does not support 8-bit weights."
            )

        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
            MarlinExperts,
        )

        # Ensure quant config is initialized
        assert self.moe_quant_config is not None, (
            "moe_quant_config must be initialized before select_gemm_impl"
        )

        w13_g_idx = (
            getattr(layer, "w13_g_idx", None) if self.quant_config.desc_act else None
        )
        w2_g_idx = (
            getattr(layer, "w2_g_idx", None) if self.quant_config.desc_act else None
        )
        w13_g_idx_sort_indices = (
            getattr(layer, "w13_g_idx_sort_indices", None)
            if self.quant_config.desc_act
            else None
        )
        w2_g_idx_sort_indices = (
            getattr(layer, "w2_g_idx_sort_indices", None)
            if self.quant_config.desc_act
            else None
        )

        # Check if using batched expert format (for Expert Parallelism)
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            # For batched format, use BatchedMarlinExperts
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None
            return BatchedMarlinExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )
        else:
            # Standard Marlin experts for GPTQ
            return MarlinExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return fused_marlin_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
            layer.w13_scales,
            layer.w2_scales,
            topk_weights,
            topk_ids,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_g_idx,
            g_idx2=layer.w2_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            is_k_full=self.is_k_full,
            input_dtype=self.input_dtype,
        )
