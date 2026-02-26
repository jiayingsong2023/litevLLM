# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from tqdm import tqdm

import vllm.envs as envs
from vllm.distributed.parallel_state import get_dp_group, is_global_first_rank
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEModularMethod
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_gemm_nt_contiguous,
)
from vllm.utils.math_utils import cdiv

def _generate_optimal_warmup_m_values(
    max_tokens: int, n: int, device: torch.device
) -> list[int]:

    # DeepGEMM's possible block sizes
    block_ms = [64, 128, 256]
    block_ns = list(range(16, min(257, n + 1), 16))
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    m_values = set()

    # Always include small cases
    m_values.update([1, 2, 4] + [i for i in range(8, 65, 8)])

    # Collect M values where different wave patterns occur
    for block_m in block_ms:
        for block_n in block_ns:
            if block_n > n:
                continue

            # Add key M boundaries for this block combination
            for wave in range(1, 11):  # Up to 10 waves
                # M where this block config transitions to next wave
                target_blocks = wave * num_sms
                m = target_blocks * block_m // cdiv(n, block_n)
                if 1 <= m <= max_tokens:
                    m_values.add(m)

            # Add block_m boundaries
            for multiple in range(1, max_tokens // block_m + 1):
                m = multiple * block_m
                if m <= max_tokens:
                    m_values.add(m)

    return sorted(m_values)

def _extract_data_from_linear_base_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    assert isinstance(m, LinearBase)
    assert isinstance(m.quant_method, Fp8LinearMethod)
    assert m.quant_method.block_quant
    assert m.quant_method.quant_config is not None

    w = m.weight
    ws = m.weight_scale_inv if hasattr(m, "weight_scale_inv") else m.weight_scale
    quant_block_size = m.quant_method.quant_config.weight_block_size

    assert isinstance(w, torch.Tensor)
    assert isinstance(ws, torch.Tensor)
    assert quant_block_size is not None
    return (w, ws, quant_block_size)

def _extract_data_from_fused_moe_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    assert isinstance(m, FusedMoE)
    w13 = m.w13_weight
    w13_s = (
        m.w13_weight_scale_inv
        if hasattr(m, "w13_weight_scale_inv")
        else m.w13_weight_scale
    )
    w2 = m.w2_weight
    w2_s = (
        m.w2_weight_scale_inv
        if hasattr(m, "w2_weight_scale_inv")
        else m.w2_weight_scale
    )
    num_topk = m.top_k

    assert isinstance(w13, torch.Tensor)
    assert isinstance(w13_s, torch.Tensor)
    assert isinstance(w2, torch.Tensor)
    assert isinstance(w2_s, torch.Tensor)
    return w13, w13_s, w2, w2_s, num_topk

def _fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:

    # FIXME: this logic is brittle and incorrect - since we
    # could use DeepGEMM with for than just Fp8LinearMethod
    block_size = get_mk_alignment_for_contiguous_layout()[0]
    if not (
        isinstance(module, LinearBase)
        and isinstance(module.quant_method, Fp8LinearMethod)
        and module.quant_method.block_quant
        and not module.quant_method.use_marlin
    ):
        return False

    w, _, block_sizes = _extract_data_from_linear_base_module(module)
    return (
        block_sizes == get_mk_alignment_for_contiguous_layout()
        and w.ndim == 2
        and w.shape[0] % block_size == 0
        and w.shape[1] % block_size == 0
    )

def _fused_moe_grouped_gemm_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    if not (envs.VLLM_USE_DEEP_GEMM and envs.VLLM_MOE_USE_DEEP_GEMM):
        return False

    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if (
        moe_quant_config is None
        or moe_quant_config.quant_dtype != torch.float8_e4m3fn
        or moe_quant_config.block_shape != get_mk_alignment_for_contiguous_layout()
    ):
        return False

    if not isinstance(module.quant_method, FusedMoEModularMethod):
        # modular kernels could invoke deep_gemm_moe_fp8
        return True

    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(
        module.quant_method.moe_mk, (DeepGemmExperts, TritonOrDeepGemmExperts)
    )

FP8_GEMM_NT_WARMUP_CACHE: set[torch.Size] = set()

def _get_fp8_gemm_nt_m_values(w: torch.Tensor, max_tokens: int) -> list[int]:
