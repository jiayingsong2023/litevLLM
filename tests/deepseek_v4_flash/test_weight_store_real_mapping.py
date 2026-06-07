from pathlib import Path

import pytest

from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_gguf_semantic_tensor_mapping() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        bindings = store.bindings

        assert bindings.token_embedding.name == "token_embd.weight"
        assert len(bindings.layers) == store.model.shape.num_layers

        # The observed target file uses canonical llama.cpp GGUF output names.
        assert "output_norm.weight" in store.model.tensors
        assert bindings.output_norm is not None
        assert bindings.output_norm.name == "output_norm.weight"
        assert "output.weight" in store.model.tensors
        assert bindings.output_head is not None
        assert bindings.output_head.name == "output.weight"

        # The target stores attention Q and output projections as A/B factors,
        # and stores K/V together in the combined attn_kv tensor.
        layer_0 = bindings.layers[0]
        assert layer_0.attention_norm is not None
        assert layer_0.attention_norm.name == "blk.0.attn_norm.weight"
        assert layer_0.attention_query_a is not None
        assert layer_0.attention_query_a.name == "blk.0.attn_q_a.weight"
        assert layer_0.attention_query_b is not None
        assert layer_0.attention_query_b.name == "blk.0.attn_q_b.weight"
        assert layer_0.attention_key_value is not None
        assert layer_0.attention_key_value.name == "blk.0.attn_kv.weight"
        assert layer_0.attention_output_a is not None
        assert layer_0.attention_output_a.name == "blk.0.attn_output_a.weight"
        assert layer_0.attention_output_b is not None
        assert layer_0.attention_output_b.name == "blk.0.attn_output_b.weight"
        assert layer_0.ffn_norm is not None
        assert layer_0.ffn_norm.name == "blk.0.ffn_norm.weight"

        grouped_expert_layers = [
            layer for layer in bindings.layers if layer.grouped_experts is not None
        ]
        assert grouped_expert_layers
        grouped_experts = grouped_expert_layers[0].grouped_experts
        assert grouped_experts is not None
        assert grouped_experts.gate.name == "blk.0.ffn_gate_exps.weight"
        assert grouped_experts.down.name == "blk.0.ffn_down_exps.weight"
        assert grouped_experts.up.name == "blk.0.ffn_up_exps.weight"

        router_layers = [layer for layer in bindings.layers if layer.router is not None]
        assert router_layers
        assert router_layers[0].router is not None
        assert router_layers[0].router.name == "blk.0.ffn_gate_inp.weight"
