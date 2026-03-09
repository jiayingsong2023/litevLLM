import subprocess
import os
import time

def run_model_test(name, model_path, batch_size, ctx_len, env_overrides=None):
    print(f"\n{'='*20} TESTING: {name} (BS={batch_size}, Ctx={ctx_len}) {'='*20}")
    base_env = os.environ.copy()
    base_env["FASTINFERENCE_GGUF_FP8"] = "1"
    base_env["FASTINFERENCE_KV_FP8"] = "1"
    base_env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    base_env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    if env_overrides:
        base_env.update(env_overrides)
    
    import_script = f"from tests.e2e_full_benchmark import benchmark_real_model; benchmark_real_model('{name}', '{model_path}', batch_size={batch_size}, context_len={ctx_len})"
    
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "-c", import_script],
            env=base_env
        )
        process.wait()
        time.sleep(2) 
    except Exception as e:
        print(f"Failed to start test for {name}: {e}")

if __name__ == "__main__":
    test_matrix = [
        ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 32, 2048, {}),
        ("Qwen3.5-9B", "models/Qwen3.5-9B-GGUF", 32, 2048, {}),
        ("Qwen3.5-35B-MoE", "models/Qwen3.5-35B-MoE-GGUF", 2, 1024, {}),
        ("DeepSeek-V2-Lite", "models/DeepSeek-V2-Lite-GGUF", 8, 4096, {}),
        ("GLM-4.7-Flash", "models/GLM-4.7-Flash-GGUF", 4, 2048, {}),
    ]
    
    for name, path, bs, ctx, env in test_matrix:
        if os.path.exists(path):
            run_model_test(name, path, bs, ctx, env)
        else:
            print(f"Skipping {name}, path not found.")
