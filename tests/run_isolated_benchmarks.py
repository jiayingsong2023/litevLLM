import subprocess
import os

def run_model_test(name, model_path, batch_size, ctx_len, env_overrides=None):
    print(f"\n{'='*20} Testing Model: {name} (BS={batch_size}, Ctx={ctx_len}) {'='*20}")
    base_env = os.environ.copy()
    if env_overrides:
        base_env.update(env_overrides)
    
    # Defaults for maximum efficiency
    base_env["FASTINFERENCE_GGUF_FP8"] = "1"
    base_env["FASTINFERENCE_KV_FP8"] = "1"
    
    import_script = f"from tests.e2e_full_benchmark import benchmark_real_model; benchmark_real_model('{name}', '{model_path}', batch_size={batch_size}, context_len={ctx_len}, env_overrides={env_overrides})"
    
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "-c", import_script],
            env=base_env
        )
        process.wait()
    except Exception as e:
        print(f"Failed to start test for {name}: {e}")

if __name__ == "__main__":
    # Performance focus: BS=16
    test_matrix = [
        ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 16, 2048, {}),
        ("Qwen3.5-9B", "models/Qwen3.5-9B-GGUF", 8, 2048, {
            "FASTINFERENCE_QWEN9_AGGRESSIVE": "1"
        }),
    ]
    
    for name, path, bs, ctx, env in test_matrix:
        if os.path.exists(path):
            run_model_test(name, path, bs, ctx, env)
        else:
            print(f"Skipping {name}, path not found: {path}")
