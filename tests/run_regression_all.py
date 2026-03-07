import os
import subprocess
import time

def run_cmd(cmd, name):
    print(f"\n{'='*20} Running {name} {'='*20}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure a clean start for each model
    process = subprocess.Popen(cmd, env=env, shell=True)
    process.wait()
    time.sleep(10) # 增加冷却时间，确保显存完全释放

# 1. DeepSeek-V2-Lite (核心性能指标)
run_cmd("uv run python tests/regression_ds_v2.py --bs 32 --context 4096", "DeepSeek-V2-Lite")

# 2. GLM-4.7-Flash
run_cmd("uv run python tests/regression_glm.py --bs 16 --context 4096", "GLM-4.7-Flash")

# 3. Qwen-3.5-9B (新增)
run_cmd("uv run python tests/regression_qwen_9b.py --bs 32 --context 4096", "Qwen-3.5-9B")

# 4. Qwen-3.5-35B
run_cmd("uv run python tests/regression_qwen_35b.py", "Qwen-3.5-35B")
