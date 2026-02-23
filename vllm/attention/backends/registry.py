# SPDX-License-Identifier: Apache-2.0
"""Attention backend registry for LiteEngine (Triton Only)"""

from enum import Enum
from typing import Any, TYPE_CHECKING
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.attention.backend import AttentionBackend

logger = init_logger(__name__)

class AttentionBackendEnum(Enum):
    """
    LiteEngine forced attention backend enum.
    Standardizes on Triton and removes all compiled backends.
    """
    TRITON_ATTN = "vllm.attention.backends.triton_attn.TritonAttentionBackend"
    CPU_ATTN = "vllm.attention.backends.cpu_attn.CPUAttentionBackend"
    
    # Stubs for compatibility with other code
    FLASH_ATTN = TRITON_ATTN
    FLASHINFER = TRITON_ATTN
    ROCM_ATTN = TRITON_ATTN
    NO_ATTENTION = "vllm.attention.backends.no_attention.NoAttentionBackend"

    def get_path(self, include_classname: bool = True) -> str:
        path = self.value
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        try:
            return resolve_obj_by_qualname(self.get_path())
        except ImportError:
            logger.warning(f"Failed to load backend {self.name}, falling back to Triton.")
            return resolve_obj_by_qualname(AttentionBackendEnum.TRITON_ATTN.value)

    def is_overridden(self) -> bool: return False
    def clear_override(self) -> None: pass

def register_backend(backend: Any, class_path: str | None = None, is_mamba: bool = False):
    return lambda x: x # No-op for LiteEngine

# Simplification for Mamba backends if needed
class MambaAttentionBackendEnum(Enum):
    LINEAR = "vllm.attention.backends.linear_attn.LinearAttentionBackend"
    def get_class(self): return resolve_obj_by_qualname(self.value)

MAMBA_TYPE_TO_BACKEND_MAP = {
    "mamba1": "MAMBA1",
    "mamba2": "MAMBA2",
    "short_conv": "SHORT_CONV",
    "linear_attention": "LINEAR",
    "gdn_attention": "GDN_ATTN",
}