# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional, Any, List
import torch
from vllm.structured_output.backend_outlines import OutlinesBackend
from vllm.structured_output.request import StructuredOutputRequest
from vllm.structured_output.backend_types import StructuredOutputGrammar

class StructuredOutputManager:
    """
    Manages structured output constraints (JSON, Regex, etc.) for FastInference.
    Uses Outlines as the primary backend for production-grade reliability.
    """
    def __init__(self, vllm_config: Any):
        self.vllm_config = vllm_config
        # We use ModelConfig directly if available
        model_config = getattr(vllm_config, "model_config", None)
        
        from vllm.model_executor.model_loader import get_tokenizer
        self.tokenizer = get_tokenizer(vllm_config.model_config)
        self.vocab_size = vllm_config.model_config.hf_config.vocab_size
        
        self.backend = OutlinesBackend(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size
        )
        
        # Cache for compiled grammars to avoid re-compilation
        # Key: (StructuredOutputOptions, spec), Value: OutlinesGrammar
        self._grammar_cache: Dict[tuple, StructuredOutputGrammar] = {}

    def get_grammar(self, request: StructuredOutputRequest) -> StructuredOutputGrammar:
        key = request.structured_output_key
        if key not in self._grammar_cache:
            option, spec = key
            self._grammar_cache[key] = self.backend.compile_grammar(option, spec)
        return self._grammar_cache[key]

    def grammar_bitmask(self, request: Optional[StructuredOutputRequest], batch_idx: int, bitmasks: torch.Tensor) -> None:
        """
        Fills the bitmask for the current request to constrain decoding.
        """
        if request is None:
            # If no constraint, allow all tokens (all 1s in bitmask)
            # In vLLM, bitmask -1 means all tokens allowed
            bitmasks[batch_idx] = -1
            return
            
        grammar = self.get_grammar(request)
        grammar.fill_bitmask(bitmasks, batch_idx)

    def should_advance(self, request: Optional[StructuredOutputRequest], token_id: int) -> bool:
        """
        Advances the FSM state after a token is accepted.
        Returns True if the token was accepted, False otherwise.
        """
        if request is None:
            return True
            
        grammar = self.get_grammar(request)
        return grammar.accept_tokens(request.params, [token_id])
