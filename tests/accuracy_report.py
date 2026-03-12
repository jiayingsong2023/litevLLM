# SPDX-License-Identifier: Apache-2.0
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gguf import GGUFConfig

# Models to test
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.models.qwen3_5 import Qwen2DecoderLayer
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.glm import GLMDecoderLayer

def run_accuracy_check(name, layer_cls, h, i, n_h, n_kv, q_cfg):
    print(f"\n[ACCURACY AUDIT] {name}")
    device = "cuda"; dtype = torch.float16
    
    class Config:
        def __init__(self):
            self.hidden_size = h; self.intermediate_size = i
            self.num_attention_heads = n_h; self.num_key_value_heads = n_kv
            self.rms_norm_eps = 1e-6; self.max_position_embeddings = 2048; self.rope_theta = 10000.0

    with torch.device(device):
        layer = layer_cls(Config(), q_cfg, "model.layers.0")
    
    def _inject(proj):
        in_s, out_s = proj.input_size, proj.output_size
        proj.qweight = nn.Parameter(torch.randint(1, 10, (out_s, in_s // 8), device=device, dtype=torch.int32), requires_grad=False)
        proj.scales = nn.Parameter(torch.ones((out_s, in_s // 128), device=device, dtype=dtype) * 0.01, requires_grad=False)
        proj.qzeros = nn.Parameter(torch.zeros((out_s, in_s // 128 // 8 + 1), device=device, dtype=torch.int32), requires_grad=False)
        proj.weight = nn.Parameter(torch.randn((out_s, in_s), device=device, dtype=dtype) * 0.01, requires_grad=False)

    for m in layer.modules():
        if isinstance(m, LiteLinear): _inject(m)
    layer.eval()
    
    # Use ROBUST inputs to avoid nan
    x = torch.ones((1, 1, h), device=device, dtype=dtype) * 0.5
    pos = torch.zeros((1, 1), device=device, dtype=torch.long)
    class MockMeta: pass
    meta = MockMeta()
    meta.slot_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    meta.block_tables = torch.zeros((1, 1), device=device, dtype=torch.int32)
    meta.seq_lens = torch.ones(1, device=device, dtype=torch.int32)
    
    kv = (torch.zeros((1, 16, n_kv, h//n_h), device=device, dtype=dtype),
          torch.zeros((1, 16, n_kv, h//n_h), device=device, dtype=dtype))

    with torch.inference_mode():
        ref_out = layer(x.clone(), pos.clone(), kv, meta)
        layer.compile_fast_dispatch()
        # Fast path requires kv[0] populated for PagedAttention check
        opt_out = layer(x.clone(), pos.clone(), kv, meta)

    cos_sim = F.cosine_similarity(ref_out.flatten(), opt_out.flatten(), dim=0).item()
    print(f"  Similarity: {cos_sim:.10f}")
    if cos_sim > 0.999:
        print("  ✅ PASS: Logic Accurate")
        return True
    else:
        print("  ❌ FAIL: Numerical Drift")
        return False

if __name__ == "__main__":
    test_matrix = [
        ("TinyLlama", LlamaDecoderLayer, 2048, 5632, 32, 4, None),
        ("Qwen3.5-9B", Qwen2DecoderLayer, 3584, 18944, 28, 4, AWQConfig()),
        ("DeepSeek-V2", DeepseekV2DecoderLayer, 5120, 12288, 128, 128, GGUFConfig()),
        ("GLM-4.7", GLMDecoderLayer, 4096, 13696, 32, 2, GGUFConfig()),
    ]
    all_pass = True
    for task in test_matrix:
        if not run_accuracy_check(*task): all_pass = False
    if not all_pass: sys.exit(1)
