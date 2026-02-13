import torch
from vllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    # Non-flash version often takes float scales
    k_scale_tensor = torch.tensor(k_scale, device=key.device, dtype=torch.float32)
    v_scale_tensor = torch.tensor(v_scale, device=value.device, dtype=torch.float32)
    
    triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale_tensor, v_scale_tensor
    )

def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    # Flash version usually takes Tensor scales
    triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )