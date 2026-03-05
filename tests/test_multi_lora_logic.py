# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from vllm import LLM, SamplingParams

def test_multi_lora_logic():
    print("=== FINAL MULTI-LORA VERIFICATION: LM_HEAD TARGET ===")
    model_path = "models/TinyLlama-1.1B-Chat-v1.0"
    
    # 1. Load Model
    llm = LLM(model=model_path)
    model = llm.model
    
    # 2. TARGET LM_HEAD: This is the most direct way to affect logits
    target_layer = model.lm_head
    print(f">>> Target Layer: {target_layer.prefix} (Input={target_layer.input_size}, Output={target_layer.output_size})")

    # 3. Inject 2 extreme adapters
    rank = 32
    # Adapter 101: Massive Positive Bias
    la1 = torch.ones((rank, target_layer.input_size), device="cuda", dtype=torch.float16) * 1.0
    lb1 = torch.ones((target_layer.output_size, rank), device="cuda", dtype=torch.float16) * 1.0
    target_layer.add_adapter(101, la1, lb1, scaling=10.0) # 10x Scaling
    
    # Adapter 102: Massive Negative Bias
    la2 = torch.ones((rank, target_layer.input_size), device="cuda", dtype=torch.float16) * 1.0
    lb2 = torch.ones((target_layer.output_size, rank), device="cuda", dtype=torch.float16) * -1.0
    target_layer.add_adapter(102, la2, lb2, scaling=10.0)
    
    print(">>> Extreme Adapters 101/102 injected into LM_HEAD.")

    # 4. Run Inference
    batch_size = 4
    lora_mapping = torch.tensor([101, 102, 0, 0], device="cuda", dtype=torch.int32)
    
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros((batch_size, 1), device="cuda", dtype=torch.long)
    num_kv_heads = llm.model_cfg.get_num_kv_heads(None)
    head_size = llm.model_cfg.get_head_size()
    kv_caches = [(torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16),
                  torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)) for _ in range(22)]
    attn_meta = {"slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32), 
                 "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32)}

    with torch.inference_mode():
        output = model(input_ids, positions, kv_caches, attn_meta, lora_mapping=lora_mapping)
    
    logits = output[:, -1, :]
    m0, m1, m2, m3 = [logits[i].mean().item() for i in range(4)]
    
    print(f"\n--- Multi-LoRA Results (LM_HEAD) ---")
    print(f"Batch 0 (Adapter 101): Mean Logit = {m0:.4f}")
    print(f"Batch 1 (Adapter 102): Mean Logit = {m1:.4f}")
    print(f"Batch 2 (Base Model):  Mean Logit = {m2:.4f}")
    print(f"Batch 3 (Base Model):  Mean Logit = {m3:.4f}")
    
    if abs(m0 - m2) > 1.0 and abs(m1 - m2) > 1.0:
        print("\n✅ SUCCESS: Multi-LoRA routing is working perfectly!")
    else:
        print("\n❌ FAILED: Still no difference. Checking model call chain...")

if __name__ == "__main__":
    test_multi_lora_logic()
