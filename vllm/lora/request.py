# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

@dataclass
class LoRARequest:
    """
    LitevLLM: Simplified LoRA request identifier.
    Used by the scheduler to group tokens by adapter.
    """
    lora_name: str
    lora_int_id: int
    lora_path: Optional[str] = None
    
    # Optional: Cache for dequantized/loaded tensors
    # In LiteLoRA, we manage the actual weights in LiteLinear
