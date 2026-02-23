# SPDX-License-Identifier: Apache-2.0
"""
LiteLinear: A unified, single-GPU linear layer supporting multiple quantization formats.
Standardizes on Triton-based execution and eliminates distributed overhead.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Any, Iterable, Tuple, Union

# Avoid top-level import of weight_utils to prevent circular imports with Attention
# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

class LiteLinear(nn.Module):
    """
    Unified, Optimized Linear layer for single-GPU LiteEngine.
    Acts as a drop-in replacement for ColumnParallelLinear, RowParallelLinear, etc.
    """
    def __init__(
        self,
        input_size: int,
        output_size: Union[int, List[int]],
        bias: bool = True,
        quant_config: Optional[Any] = None,
        params_dtype: torch.dtype = torch.float16,
        prefix: str = "",
        # Compatibility arguments for Parallel Linear Layers
        gather_output: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        disable_tp: bool = False,
        reduce_results: bool = True,
        **kwargs
    ):
        super().__init__()
        
        if isinstance(output_size, list):
            self.output_partition_sizes = output_size
            self.output_size = sum(output_size)
        else:
            self.output_partition_sizes = [output_size]
            self.output_size = output_size
            
        self.input_size = input_size
        self.prefix = prefix
        self.params_dtype = params_dtype
        self.skip_bias_add = skip_bias_add
        self.return_bias = return_bias
        
        if quant_config:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        else:
            self.quant_method = UnquantizedLiteMethod()

        # Extra attributes for weight loading
        extra_weight_attrs = {"weight_loader": self.weight_loader}
        
        self.quant_method.create_weights(
            self,
            input_size_per_partition=input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=input_size,
            output_size=self.output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs
        )
        
        if hasattr(self, "weight") and not hasattr(self.weight, "weight_loader"):
            set_weight_attrs(self.weight, extra_weight_attrs)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, extra_weight_attrs)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: Any):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        
        shard_id = None
        if isinstance(loaded_weight, tuple):
            loaded_weight, shard_id = loaded_weight

        if shard_id is None:
            default_weight_loader(param, loaded_weight)
        else:
            # Safer offset calculation based on actual partition sizes
            offset = 0
            if shard_id == "q" or shard_id == "gate" or shard_id == 0:
                offset = 0
            elif shard_id == "k" or shard_id == "up" or shard_id == 1:
                if len(self.output_partition_sizes) > 1:
                    offset = self.output_partition_sizes[0]
            elif shard_id == "v" or shard_id == 2:
                if len(self.output_partition_sizes) > 2:
                    offset = self.output_partition_sizes[0] + self.output_partition_sizes[1]
            elif isinstance(shard_id, int):
                if shard_id < len(self.output_partition_sizes):
                    offset = sum(self.output_partition_sizes[:shard_id])
            
            shard_size = loaded_weight.shape[0]
            # Verify we don't overflow
            if offset + shard_size > param.data.shape[0]:
                shard_size = param.data.shape[0] - offset
                if shard_size <= 0: return

            param.data.narrow(0, offset, shard_size).copy_(loaded_weight[:shard_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.quant_method.apply(self, x)
        
        if self.bias is not None:
            if self.skip_bias_add:
                return out, self.bias
            else:
                return out + self.bias
        
        if self.return_bias:
            return out, None
            
        return out

class UnquantizedLiteMethod:
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs):
        weight = nn.Parameter(torch.empty(sum(output_partition_sizes), input_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer, x):
        return torch.matmul(x, layer.weight.t())