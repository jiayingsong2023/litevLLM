# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LiteInferenceConfig:
    # KV Cache Configuration
    kv_type: str  # 'turbo_int4', 'fp8', 'fp16'
    k_scale: float
    v_scale: float
    
    # Model Optimization Configuration
    # 0: No fusion
    # 1: Basic fusion (AB Gemm)
    # 2: Aggressive fusion (GateUp, KV, etc.)
    fusion_level: int
    
    # Engine Limits
    block_size: int
    max_model_len: Optional[int]
    max_active_requests: int
    
    # Qwen-specific stability
    use_prompt_guard: bool

    @classmethod
    def from_env(cls) -> "LiteInferenceConfig":
        # Resolve KV Type (Defaulting to fp8 for balanced stability and memory)
        kv_type = os.environ.get("FASTINFERENCE_KV_TYPE", "auto")
        if kv_type == "auto":
            kv_type = "fp8"
        
        # Resolve Fusion Level (Consolidating QWEN35_FUSED_*)
        # Default to level 2 (Aggressive) if not set
        fusion_level_env = os.environ.get("FASTINFERENCE_FUSION_LEVEL")
        if fusion_level_env is not None:
            fusion_level = int(fusion_level_env)
        else:
            fusion_level = 2
        
        block_size_env = os.environ.get("FASTINFERENCE_BLOCK_SIZE", "").strip()
        block_size = int(block_size_env) if block_size_env else 16
        if block_size not in [16, 32, 64]:
            print(f">>>> [Warning] Unsupported block_size {block_size}. Defaulting to 16.")
            block_size = 16
        
        max_model_len_env = os.environ.get("FASTINFERENCE_KV_MAX_MODEL_LEN", "").strip()
        max_model_len = int(max_model_len_env) if max_model_len_env else None
            
        max_active_requests = int(os.environ.get("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "4"))
        
        return cls(
            kv_type=kv_type,
            k_scale=float(os.environ.get("FASTINFERENCE_K_SCALE", "1.0")),
            v_scale=float(os.environ.get("FASTINFERENCE_V_SCALE", "1.0")),
            fusion_level=fusion_level,
            block_size=block_size,
            max_model_len=max_model_len,
            max_active_requests=max_active_requests,
            use_prompt_guard=os.environ.get("FASTINFERENCE_QWEN35_PROMPT_GUARD", "1") == "1"
        )
