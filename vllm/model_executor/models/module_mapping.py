# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional

class LiteModelMapping:
    """
    Central registry for mapping LitevLLM standardized layer names
    to original HuggingFace/GGUF tensor names.
    """
    # Llama/Qwen Standard Mapping
    _LLAMA_MAPPING = {
        "self_attn.qkv_proj": ["attn_q", "attn_k", "attn_v", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "self_attn.o_proj": ["attn_output", "self_attn.o_proj"],
        "mlp.gate_up_proj": ["ffn_gate", "ffn_up", "mlp.gate_proj", "mlp.up_proj"],
        "mlp.down_proj": ["ffn_down", "mlp.down_proj"],
        "input_layernorm": ["attn_norm", "input_layernorm"],
        "post_attention_layernorm": ["ffn_norm", "post_attention_layernorm"],
        "lm_head": ["output", "lm_head"]
    }

    # DeepSeek Mapping (MLA)
    _DEEPSEEK_MAPPING = {
        "self_attn.qkv_proj": ["attn_q_a_proj", "attn_q_b_proj", "attn_kv_a_proj", "attn_kv_b_proj"],
        "mlp": ["feed_forward", "mlp"], # MoE specific
    }

    # GLM Mapping
    _GLM_MAPPING = {
        "self_attn.qkv_proj": ["query_key_value"],
        "self_attn.o_proj": ["dense"],
        "mlp.gate_up_proj": ["dense_h_to_4h"],
        "mlp.down_proj": ["dense_4h_to_h"],
    }

    @staticmethod
    def get_mapping(arch: str) -> Dict[str, List[str]]:
        if "Qwen" in arch or "Llama" in arch:
            return LiteModelMapping._LLAMA_MAPPING
        if "Deepseek" in arch:
            return {**LiteModelMapping._LLAMA_MAPPING, **LiteModelMapping._DEEPSEEK_MAPPING}
        if "Glm" in arch:
            return {**LiteModelMapping._LLAMA_MAPPING, **LiteModelMapping._GLM_MAPPING}
        return LiteModelMapping._LLAMA_MAPPING

    @staticmethod
    def resolve_tensor_name(prefix: str, arch: str) -> List[str]:
        """
        Given a LitevLLM prefix (e.g. 'blk.0.self_attn.qkv_proj'),
        returns a list of potential original tensor suffixes to look for.
        """
        mapping = LiteModelMapping.get_mapping(arch)
        # Extract the suffix (e.g. self_attn.qkv_proj)
        parts = prefix.split('.')
        # Heuristic: try to match known suffixes
        for key in mapping:
            if prefix.endswith(key):
                return mapping[key]
        return []
