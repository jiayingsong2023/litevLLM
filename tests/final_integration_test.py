import sys
import torch
import unittest
import os
from unittest.mock import MagicMock

# 1. Mock heavy modules
sys.modules["vllm.model_executor.model_loader"] = MagicMock()
sys.modules["vllm.engine.loadtime_policy"] = MagicMock()

from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams

class FinalIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        # Mock Configs
        self.max_len = 128
        self.max_reqs = 4
        
        class FakeModelConfig:
            model = "test-model"
            tokenizer = "test-model"
            trust_remote_code = True
            dtype = "float16"
            max_model_len = 128
            def get_num_layers(self, _): return 2
            def get_num_kv_heads(self, _): return 4
            def get_head_size(self): return 64
            def get_total_num_kv_heads(self): return 4
            def get_max_model_len(self): return 128

        class FakePolicy:
            max_active_requests = 4
            max_tokens_cap = 128
            chunked_prefill_size = 16 # Small chunk for testing

        self.vllm_config = MagicMock()
        self.vllm_config.model_config = FakeModelConfig()
        sys.modules["vllm.engine.loadtime_policy"].select_loadtime_policy.return_value = FakePolicy()
        sys.modules["vllm.model_executor.model_loader"].get_model.return_value = torch.nn.Linear(1, 1).cuda()

    def _run_engine_test(self, kv_fp8=False):
        if kv_fp8:
            os.environ["FASTINFERENCE_KV_FP8"] = "1"
        else:
            os.environ["FASTINFERENCE_KV_FP8"] = "0"
            
        engine = LiteEngine(self.vllm_config)
        
        # Verify KV Cache Dtype
        expected_dtype = torch.float8_e4m3fn if kv_fp8 else torch.float16
        self.assertEqual(engine.kv_caches[0][0].dtype, expected_dtype)
        
        # Instrumented Mock Model
        call_log = []
        class MockModel(torch.nn.Module):
            def forward(self, input_ids, positions, kv_caches, attn_metadata):
                call_log.append({
                    "shape": input_ids.shape,
                    "is_prefill": attn_metadata["is_prefill"],
                    "kv_start": attn_metadata["kv_start_indices"].tolist()
                })
                # Return random logits
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 32000, device="cuda")
        
        engine.model = MockModel().cuda()
        
        # Mock Tokenizer
        class MockTokenizer:
            def encode(self, text): return [1] * len(text.split())
            def decode(self, ids): return f"token-{ids[-1]}"
            eos_token_id = 999
        engine.tokenizer = MockTokenizer()

        # Test Case 1: Multiple requests (Continuous Batching)
        engine.add_request("req1", "word " * 4, SamplingParams(max_tokens=3)) 
        engine.add_request("req2", "word " * 19, SamplingParams(max_tokens=2))

        print(f"--- Testing {'FP8' if kv_fp8 else 'FP16'} Engine ---")
        
        # Step 1: Process req1 prefill (small)
        engine.step()
        self.assertEqual(call_log[-1]["shape"], (1, 4))
        self.assertTrue(call_log[-1]["is_prefill"])

        # Step 2: Process req2 prefill CHUNK 1 (16 tokens)
        engine.step()
        self.assertEqual(call_log[-1]["shape"], (1, 16))
        self.assertTrue(call_log[-1]["is_prefill"])
        self.assertEqual(call_log[-1]["kv_start"], [0])

        # Step 3: Process req2 prefill CHUNK 2 (remaining 3 tokens)
        engine.step()
        self.assertEqual(call_log[-1]["shape"], (1, 3))
        self.assertTrue(call_log[-1]["is_prefill"])
        self.assertEqual(call_log[-1]["kv_start"], [16]) # Verify chunk offset

        # Step 4: Now both are in Decode mode. Should BATCH them.
        engine.step()
        self.assertEqual(call_log[-1]["shape"], (2, 1)) # BATCHED!
        self.assertFalse(call_log[-1]["is_prefill"])
        
        # Step 5: Continue decoding
        engine.step()
        
        # Verify engine becomes empty
        self.assertEqual(len(engine._running_ids), 0)
        print("Success: Continuous Batching + Chunked Prefill verified.")

    def test_fp16_engine(self):
        self._run_engine_test(kv_fp8=False)

    def test_fp8_engine(self):
        self._run_engine_test(kv_fp8=True)

if __name__ == "__main__":
    unittest.main()
