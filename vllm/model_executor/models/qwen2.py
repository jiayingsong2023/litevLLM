# SPDX-License-Identifier: Apache-2.0
"""
Qwen2: LitevLLM Optimized Implementation.
Qwen2 shares the same architecture as Llama, so we alias the core logic.
"""

from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel


class Qwen2Model(LlamaModel):
    """Qwen2 backbone, identical to Llama."""

    pass


class Qwen2ForCausalLM(LlamaForCausalLM):
    """Qwen2 causal model, identical to Llama."""

    pass
