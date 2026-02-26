# SPDX-License-Identifier: Apache-2.0
from typing import Optional

class AttentionConfig:
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        block_size: int = 16,
        is_causal: bool = True,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.is_causal = is_causal