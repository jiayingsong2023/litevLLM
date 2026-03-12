# SPDX-License-Identifier: Apache-2.0
import torch
import unittest
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.models.llama import LlamaModel

class TestLitevLLM(unittest.TestCase):
    def test_lite_linear_init(self):
        """Verify LiteLinear can be initialized and run forward."""
        device = "cuda"
        # Correct init for new LiteLinear signature
        linear = LiteLinear(128, 256, bias=True)
        linear.to(device)
        x = torch.randn(8, 128, device=device)
        out = linear(x)
        self.assertEqual(out.shape, (8, 256))
        self.assertEqual(out.device.type, "cuda")

    def test_llama_structure(self):
        """Verify LlamaModel structure with standardized prefix."""
        class DummyHFConfig:
            def __init__(self):
                self.hidden_size = 128; self.intermediate_size = 256
                self.num_attention_heads = 4; self.num_key_value_heads = 4
                self.num_hidden_layers = 2; self.rms_norm_eps = 1e-6
                self.vocab_size = 1000; self.max_position_embeddings = 2048; self.rope_theta = 10000.0
        
        # Test loading with the new Mirror refactoring
        model = LlamaModel(DummyHFConfig(), quant_config=None, prefix="model")
        self.assertTrue(hasattr(model, "layers"))
        self.assertEqual(len(model.layers), 2)
        # Verify official naming chain
        self.assertTrue(hasattr(model.layers[0], "self_attn"))
        self.assertTrue(hasattr(model.layers[0].self_attn, "q_proj"))

if __name__ == "__main__":
    unittest.main()
