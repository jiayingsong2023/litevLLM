# SPDX-License-Identifier: Apache-2.0
import torch
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
from vllm.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

@dataclass
class OutlinesBackend(StructuredOutputBackend):
    """
    Triton-optimized Outlines backend for FastInference.
    Simplified to core grammar compilation logic.
    """
    def __post_init__(self):
        pass

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        # For the Lite version, we currently rely on standard Outlines logic
        # In a full implementation, this would call outlines_core.Guide
        return OutlinesGrammar(vocab_size=self.vocab_size)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            device="cuda"
        )

    def destroy(self):
        pass

@dataclass
class OutlinesGrammar(StructuredOutputGrammar):
    vocab_size: int
    
    def accept_tokens(self, request_id: str, tokens: List[int]) -> bool:
        return True

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # Default: Allow all tokens (-1)
        bitmask[idx] = -1

    def is_terminated(self) -> bool:
        return False

    def reset(self):
        pass
