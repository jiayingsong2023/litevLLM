# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, Tuple
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.kernels.triton.moe_gemm import index_aware_linear

class LiteLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix
        
        # --- Multi-adapter LiteLoRA Pool ---
        # Key: lora_int_id, Value: (lora_a, lora_b, scaling)
        self.lora_adapters: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}

        if self.quant_config is None:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            self.quant_config.init_layer(self)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def add_adapter(self, lora_id: int, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
        """注册或更新一个 LoRA 适配器权重"""
        self.lora_adapters[lora_id] = (lora_a, lora_b, scaling)

    def remove_adapter(self, lora_id: int):
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]

    def forward(self, x: torch.Tensor, lora_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        lora_mapping: [num_tokens] 存储每个 token 对应的 lora_id
        """
        # 1. Base Weight Path (Triton Cached)
        if self.quant_config is None:
            res = torch.nn.functional.linear(x, self.weight, self.bias)
        else:
            res = self.quant_config.apply(self, x)
            
        # 2. Multi-adapter LiteLoRA Path
        if not self.lora_adapters or lora_mapping is None:
            return res

        # 优化：通过 Index-aware GEMM 并行处理多个适配器
        # 我们对 lora_mapping 进行分组处理
        unique_ids = lora_mapping.unique()
        for lora_id in unique_ids:
            lora_id_item = lora_id.item()
            if lora_id_item == 0: # 0 通常代表 Base Model
                continue
                
            if lora_id_item in self.lora_adapters:
                la, lb, scaling = self.lora_adapters[lora_id_item]
                
                # 获取属于该适配器的 token 索引
                indices = (lora_mapping == lora_id).nonzero().flatten()
                
                # --- 复用 MoE 优化的 Index-aware 逻辑 ---
                # A 路径: X @ la.T
                # 我们需要一个临时缓冲区存放中间结果
                # 为了 Lite 版本的简洁，此处使用 index_select，但在生产中我们会使用融合 Kernel
                tokens = x.index_select(0, indices)
                lora_res = torch.nn.functional.linear(tokens, la)
                lora_res = torch.nn.functional.linear(lora_res, lb)
                
                # 原地累加回结果矩阵
                res.index_add_(0, indices, lora_res * scaling)
                
        return res

    def load_weights(self, weights_iter):
        if self.quant_config is None:
            for name, loaded_weight in weights_iter:
                if "weight" in name:
                    self.weight.data.copy_(loaded_weight)
                elif "bias" in name and self.bias is not None:
                    self.bias.data.copy_(loaded_weight)
        else:
            self.quant_config.load_weights(self, weights_iter)
