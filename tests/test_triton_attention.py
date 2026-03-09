
import torch
import unittest
from vllm.model_executor.models.llama import LlamaAttention

class MockConfig:
    hidden_size = 64
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 16

class TestTritonAttention(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.config = MockConfig()
        self.attn = LlamaAttention(self.config, layer_id=0, quant_config=None, prefix="test").to(self.device).half()

    def test_triton_decode(self):
        bsz = 4 # 测试 Batching
        seq_len = 1
        max_active = 16
        max_len = 128
        k_cache = torch.zeros(max_active, max_len, self.config.num_key_value_heads, self.config.head_dim, device=self.device, dtype=torch.float16)
        v_cache = torch.zeros(max_active, max_len, self.config.num_key_value_heads, self.config.head_dim, device=self.device, dtype=torch.float16)
        kv_cache = (k_cache, v_cache)
        
        # 模拟历史
        hist_len = 10
        k_cache[:, :hist_len] = torch.randn_like(k_cache[:, :hist_len])
        v_cache[:, :hist_len] = torch.randn_like(v_cache[:, :hist_len])
        
        hidden_states = torch.randn(bsz, seq_len, self.config.hidden_size, device=self.device, dtype=torch.float16)
        attn_metadata = {
            "slot_mapping": torch.tensor([0, 2, 4, 6], device=self.device, dtype=torch.long), # 非连续 Slot
            "seq_lens": torch.tensor([hist_len+1]*bsz, device=self.device, dtype=torch.int32),
            "kv_start_indices": torch.tensor([hist_len]*bsz, device=self.device, dtype=torch.int32)
        }
        
        # 运行 Triton 路径
        output = self.attn(hidden_states, None, kv_cache, attn_metadata)
        
        self.assertEqual(output.shape, (bsz, seq_len, self.config.hidden_size))
        print("Triton Decode Test Passed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
