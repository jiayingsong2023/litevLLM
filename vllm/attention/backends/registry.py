# SPDX-License-Identifier: Apache-2.0
from enum import Enum, auto

class AttentionBackendEnum(Enum):
    TRITON_ATTN = auto()
    FLASH_ATTN = auto()
    XFORMERS = auto()
    ROCM_FLASH = auto()
    TORCH_SDPA = auto()
    OPENVINO = auto()
    FLASHINFER = auto()
    PALLADIUM = auto() 
    # Mamba backends
    MAMBA_INNER = auto()
    MAMBA_OUTER = auto()
    
    def get_name(self) -> str:
        return self.name

    def is_mla(self) -> bool:
        return False
