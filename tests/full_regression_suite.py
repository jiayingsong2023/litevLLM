# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import sys
import time
import re

# Threshold for performance pass (as percentage of baseline)
PERF_THRESHOLD = 0.85 

# Model Test Configurations
# Format: (Name, Path, BatchSize, ContextLen, BaselineTPS)
REGRESSION_MATRIX = [
    ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 16, 2048, 420.0),
    ("Qwen3.5-9B", "models/Qwen3.5-9B-GGUF", 8, 2048, 170.0),
    ("Qwen3.5-35B-MoE", "models/Qwen3.5-35B-MoE-GGUF", 2, 1024, 115.0),
    ("DeepSeek-V2-Lite", "models/DeepSeek-V2-Lite-GGUF", 4, 4096, 600.0),
]

def run_single_regression(name, path, bs, ctx, baseline):
    print(f"\n[REGRESSION] Testing {name}...")
    env = os.environ.copy()
    env["FASTINFERENCE_KV_FP8"] = "1"
    env["FASTINFERENCE_GGUF_FP8"] = "1"
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Python script to execute the benchmark
    cmd = f"from tests.e2e_full_benchmark import benchmark_real_model; benchmark_real_model('{name}', '{path}', {bs}, {ctx})"
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "-c", cmd],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        output = ""
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None: break
            if line:
                output += line
                if "RESULT:" in line or "Failed" in line:
                    print(f"  {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"❌ CRASH: {name} exited with code {process.returncode}")
            return False, 0.0

        # Extract TPS
        match = re.search(r"Throughput=([\d\.]+) tokens/sec", output)
        if match:
            tps = float(match.group(1))
            ratio = tps / baseline
            status = "✅ PASS" if ratio >= PERF_THRESHOLD else "⚠️ SLOW"
            print(f"  Status: {status} ({tps:.2f} TPS, {ratio*100:.1f}% of baseline)")
            return ratio >= PERF_THRESHOLD, tps
        else:
            print(f"❌ ERROR: Could not parse TPS for {name}")
            return False, 0.0
            
    except Exception as e:
        print(f"❌ EXCEPTION: {name} failed with {e}")
        return False, 0.0

if __name__ == "__main__":
    print("="*80)
    print("LitevLLM v2.0 - FULL PERFORMANCE REGRESSION SUITE")
    print("="*80)
    
    results = []
    overall_success = True
    
    for name, path, bs, ctx, baseline in REGRESSION_MATRIX:
        if not os.path.exists(path):
            print(f"\n[SKIP] {name} (Path not found: {path})")
            continue
            
        success, tps = run_single_regression(name, path, bs, ctx, baseline)
        results.append((name, success, tps, baseline))
        if not success:
            overall_success = False
            
    print("\n" + "="*80)
    print("FINAL REGRESSION SUMMARY")
    print("-" * 80)
    print(f"{'Model':<20} | {'Status':<10} | {'Actual':<10} | {'Baseline':<10}")
    for name, success, tps, baseline in results:
        status_str = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<20} | {status_str:<10} | {tps:<10.2f} | {baseline:<10.2f}")
    
    print("="*80)
    if overall_success:
        print("🎉 ALL SYSTEMS GO. READY FOR SYNC.")
        sys.exit(0)
    else:
        print("🛑 REGRESSION FAILED. CHECK LOGS BEFORE SYNC.")
        sys.exit(1)
