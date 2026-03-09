
import subprocess
import os
import re

def run_bench(name, model_path, bs, ctx):
    env = os.environ.copy()
    env["FASTINFERENCE_GGUF_FP8"] = "1"
    env["FASTINFERENCE_KV_FP8"] = "1"
    
    cmd = f"from tests.e2e_full_benchmark import benchmark_real_model; benchmark_real_model('{name}', '{model_path}', batch_size={bs}, context_len={ctx})"
    
    try:
        result = subprocess.check_output(["uv", "run", "python", "-c", cmd], env=env, stderr=subprocess.STDOUT).decode()
        match = re.search(r"Throughput=([\d\.]+) tokens/sec", result)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error running {name}: {e}")
    return 0.0

if __name__ == "__main__":
    baselines = {
        "TinyLlama-1.1B": 602.5,
        "Qwen3.5-9B": 240.6,
        "Qwen3.5-35B-MoE": 3.5
    }
    
    test_matrix = [
        ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 32, 128),
        ("Qwen3.5-9B", "models/Qwen3.5-9B-GGUF", 32, 128),
        ("Qwen3.5-35B-MoE", "models/Qwen3.5-35B-MoE-GGUF", 1, 128),
    ]
    
    print(f"{'Model':<20} | {'Baseline':<10} | {'Current':<10} | {'Status':<10}")
    print("-" * 60)
    
    for name, path, bs, ctx in test_matrix:
        if not os.path.exists(path):
            print(f"{name:<20} | {'N/A':<10} | {'SKIP':<10} | {'MISSING'}")
            continue
            
        current_tps = run_bench(name, path, bs, ctx)
        baseline = baselines[name]
        diff = (current_tps / baseline) * 100 if baseline > 0 else 0
        status = "✅ PASS" if diff > 80 else "⚠️ SLOW" if diff > 50 else "❌ FAIL"
        
        print(f"{name:<20} | {baseline:<10.2f} | {current_tps:<10.2f} | {status} ({diff:.1f}%)")
