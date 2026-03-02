# SPDX-License-Identifier: Apache-2.0
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any
import torch

class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()

StructuredOutputKey = Tuple[StructuredOutputOptions, str]

class StructuredOutputGrammar(ABC):
    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: List[int]) -> bool:
        pass

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

@dataclass
class StructuredOutputBackend(ABC):
    vllm_config: Any
    tokenizer: Any
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        pass

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        pass

    @abstractmethod
    def destroy(self):
        pass
