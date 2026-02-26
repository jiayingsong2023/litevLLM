# SPDX-License-Identifier: Apache-2.0
import torch
import unittest
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.layers.lite_linear import LiteLinear

class TestLitevLLM(unittest.TestCase):
    def test_lite_linear_init(self):
        """Verify LiteLinear can be initialized without distributed env."""
        linear = LiteLinear(128, 256, bias=True)
        x = torch.randn(1, 128, device="cuda")
        y = linear(x)
        self.assertEqual(y.shape, (1, 256))
        print("LiteLinear Init & Forward: SUCCESS")

    def test_llama_structure(self):
        """Verify LlamaModel structure with LiteLinear."""
        # Using a dummy config-like object
        class DummyConfig:
            def __init__(self):
                self.hidden_size = 128
                self.intermediate_size = 256
                self.num_hidden_layers = 2
                self.num_attention_heads = 4
                self.num_key_value_heads = 2
                self.rms_norm_eps = 1e-6
                self.vocab_size = 1000
                self.max_position_embeddings = 512
                self.rope_theta = 10000
                self.head_dim = 32

        class DummyVllmConfig:
            def __init__(self):
                self.model_config = type('obj', (object,), {'hf_config': DummyConfig()})
                self.quant_config = None

        model = LlamaModel(DummyVllmConfig())
        self.assertEqual(len(model.layers), 2)
        print("LlamaModel Structure: SUCCESS")

if __name__ == "__main__":
    unittest.main()
