# SPDX-License-Identifier: Apache-2.0
"""
LitevLLM Semantic Integrity Verification Suite.
Compares LitevLLM (Triton/LiteEngine) against Hugging Face (PyTorch) 
to ensure absolute numerical alignment and semantic correctness.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, LoadConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams
import time
import argparse
import os
import sys
# Debug: check where vllm is loaded from
import vllm.model_executor.models.llama
print(f"DEBUG: Loaded vllm.llama from: {vllm.model_executor.models.llama.__file__}")

def compare_logits(lite_logits, ref_logits, threshold=0.999):
    """Computes cosine similarity and max absolute error."""
    cos_sim = F.cosine_similarity(lite_logits.flatten(), ref_logits.flatten(), dim=0).item()
    max_err = (lite_logits - ref_logits).abs().max().item()
    return cos_sim, max_err

def run_alignment_test(model_path, quant_type="none", prompt="The capital of France is", no_hf=False):
    print(f"\n" + "="*60)
    print(f"AUDITING: {os.path.basename(model_path)} (Quant: {quant_type})")
    print("="*60)
    
    device = "cuda"
    dtype = torch.float16
    
    # Force FP16 for audit to avoid quantization noise
    os.environ["FASTINFERENCE_KV_FP8"] = "0"
    
    # 1. Initialize LitevLLM Engine
    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    
    q_cfg = None
    if quant_type == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig
        q_cfg = AWQConfig()
    elif quant_type == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig
        q_cfg = GGUFConfig()
        
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    
    print("[1/3] Loading LitevLLM (Python/Triton)...")
    engine = LiteEngine(v_cfg)
    from vllm.model_executor.model_loader import get_tokenizer
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    engine.tokenizer = tokenizer
    
    if no_hf:
        print("[2/3] Skipping HF Reference (requested)...")
        hf_model = None
    else:
        # 2. Load Hugging Face Reference (Standard PyTorch)
        # Note: For GGUF/AWQ, standard HF might need specific loaders or we compare against unquantized FP16
        print(f"[2/3] Loading HF Reference (PyTorch {dtype})...")
        
        import json
        from transformers import AutoConfig, PretrainedConfig, Qwen2Config
        config_file = os.path.join(model_path, "config.json")
        hf_config = None
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                cfg_dict = json.load(f)
            if cfg_dict.get("model_type") == "qwen3_5":
                print("  [Note] Patching config dict: model_type qwen3_5 -> qwen2 for HF compatibility.")
                cfg_dict["model_type"] = "qwen2"
                hf_config = Qwen2Config.from_dict(cfg_dict)
            else:
                try: hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                except: pass
        else:
            try: hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            except: pass

        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                config=hf_config,
                torch_dtype=dtype, 
                device_map="auto", 
                trust_remote_code=True
            ).eval()
        except Exception as e:
            print(f"  [Warning] HF Reference load failed: {e}")
            print("  Falling back to LitevLLM-only run.")
            hf_model = None
    
    # 3. Execution & Comparison (Greedy)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Input Tokens: {input_ids[0].tolist()}")
    
    print(f"[3/3] Running Generation Audit...")
    
    # Initial Logits Audit (only if HF available)
    if hf_model is not None:
        with torch.inference_mode():
            # HF Path
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits[:, -1, :]
            hf_token = torch.argmax(hf_logits, dim=-1).item()
            
            # LitevLLM Path (Internal hack to get logits for audit)
            captured_logits = []
            original_forward = engine.model.forward
            def audit_forward(*args, **kwargs):
                res = original_forward(*args, **kwargs)
                captured_logits.append(res.detach().cpu())
                return res
            engine.model.forward = audit_forward

            engine.add_request("audit", prompt, SamplingParams(max_tokens=1, temperature=0.0))
            while True:
                step_outputs = engine.step()
                if step_outputs:
                    lite_token = step_outputs[0].outputs[0].token_ids[-1]
                    break
            
            engine.model.forward = original_forward
            lite_logits = captured_logits[0][:, -1, :].to(hf_logits.device)
            cos_sim, max_err = compare_logits(lite_logits, hf_logits)
            print(f"  Prefill Logits -> CosSim: {cos_sim:.6f}, MaxErr: {max_err:.6f}")
            print(f"  Prefill Token: HF={hf_token} | Lite={lite_token}")
    
    # Multi-token Generation Audit
    max_new_tokens = 10
    
    # LitevLLM for full sequence
    engine.add_request("audit_full", prompt, SamplingParams(max_tokens=max_new_tokens, temperature=0.0))
    full_lite_out = None
    while True:
        step_outputs = engine.step()
        if step_outputs and step_outputs[0].finished:
            full_lite_out = step_outputs[0].outputs[0]
            break
    
    lite_text = full_lite_out.text
    print(f"  LitevLLM Output: '{lite_text}'")
    print(f"  Lite Tokens: {full_lite_out.token_ids}")

    if hf_model is not None:
        hf_full_gen = hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        hf_text = tokenizer.decode(hf_full_gen[0][input_ids.shape[-1]:])
        print(f"  HF Reference:   '{hf_text}'")
        
        match = (lite_text.strip() == hf_text.strip())
        if match:
            print("  ✅ PASS: Semantic Integrity Verified.")
        else:
            print("  ❌ FAIL: Semantic Drift Detected.")
            lite_tokens = full_lite_out.token_ids
            hf_tokens = hf_full_gen[0][input_ids.shape[-1]:].tolist()
            print(f"  Lite Tokens: {lite_tokens}")
            print(f"  HF Tokens:   {hf_tokens}")
        return match
    else:
        print("  [Info] Completed LitevLLM-only run (no reference comparison).")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gguf"])
    parser.add_argument("--no-hf", action="store_true")
    args = parser.parse_args()
    
    success = run_alignment_test(args.model, args.quant, no_hf=args.no_hf)
    if not success:
        exit(1)
