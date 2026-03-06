# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
import gguf
from contextlib import contextmanager
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache


def _is_gguf_dir(model_path: str) -> bool:
    return os.path.isdir(model_path) and any(f.endswith(".gguf") for f in os.listdir(model_path))


def _assert_quant_weights_ready(model) -> None:
    missing_prefixes = []
    checked = 0
    for _, module in model.named_modules():
        if hasattr(module, "quant_config") and module.quant_config is not None:
            checked += 1
            has_packed_gguf = (
                getattr(module, "qweight", None) is not None
                and getattr(module, "gguf_quant_type", None) is not None
            )
            has_legacy_gguf = (
                getattr(module, "qweight", None) is not None
                and getattr(module, "scales", None) is not None
            )
            if not has_packed_gguf and not has_legacy_gguf:
                missing_prefixes.append(getattr(module, "prefix", module.__class__.__name__))

    if checked == 0:
        return

    if missing_prefixes:
        preview = ", ".join(missing_prefixes[:8])
        if len(missing_prefixes) > 8:
            preview += f", ... (+{len(missing_prefixes) - 8} more)"
        raise RuntimeError(
            "Quantized weights are incomplete. "
            f"Missing qweight/scales on {len(missing_prefixes)} modules: {preview}"
        )


def _as_int_value(value):
    if isinstance(value, list):
        if not value:
            return 1
        return max(1, int(max(value)))
    return int(value)


def _infer_model_dtype(hf_config) -> torch.dtype:
    raw_dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
    if isinstance(raw_dtype, torch.dtype):
        return raw_dtype
    if isinstance(raw_dtype, str):
        normalized = raw_dtype.lower()
        if "bfloat16" in normalized or normalized == "bf16":
            return torch.bfloat16
        if "float32" in normalized or normalized == "fp32":
            return torch.float32
    return torch.float16


@contextmanager
def _temporary_env(overrides: dict[str, str] | None):
    if not overrides:
        yield
        return
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def load_real_config(model_path):
    """Loads configuration from a real model directory."""
    if os.path.isdir(model_path):
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            # For Qwen3.5 and others with nested configs
            with open(config_file, "r") as f:
                data = json.load(f)
            
            # Simple wrapper to mimic HF config object
            class RealConfig:
                def __init__(self, d, architectures):
                    self.__dict__.update(d)
                    self.architectures = architectures
            
            text_config = data.get("text_config", data)
            archs = data.get("architectures", [])
            return RealConfig(text_config, archs)

        gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
        if gguf_files:
            gguf_file = os.path.join(model_path, gguf_files[0])
            reader = gguf.GGUFReader(gguf_file)
            architecture = str(reader.get_field("general.architecture").contents())

            def read_field_int(field_name: str, default: int) -> int:
                field = reader.get_field(field_name)
                if field is None:
                    return default
                return _as_int_value(field.contents())

            def read_field_raw(field_name: str, default):
                field = reader.get_field(field_name)
                if field is None:
                    return default
                return field.contents()

            class GGUFConfigFallback:
                pass

            fallback = GGUFConfigFallback()
            fallback.hidden_size = read_field_int(f"{architecture}.embedding_length", 4096)
            fallback.num_hidden_layers = read_field_int(f"{architecture}.block_count", 32)
            fallback.num_attention_heads = read_field_int(
                f"{architecture}.attention.head_count", 32
            )
            fallback.num_key_value_heads = read_field_int(
                f"{architecture}.attention.head_count_kv",
                fallback.num_attention_heads,
            )
            fallback.intermediate_size = read_field_int(
                f"{architecture}.feed_forward_length", 11008
            )
            fallback.max_position_embeddings = read_field_int(
                f"{architecture}.context_length", 32768
            )
            fallback.vocab_size = read_field_int(f"{architecture}.vocab_size", 32000)
            fallback.rms_norm_eps = 1e-6
            if architecture == "kimi-linear":
                fallback.architectures = ["KimiLinearForCausalLM"]
                kv_pattern = read_field_raw(f"{architecture}.attention.head_count_kv", [])
                if isinstance(kv_pattern, list):
                    full_layers = [idx for idx, value in enumerate(kv_pattern) if int(value) > 0]
                else:
                    full_layers = []
                fallback.linear_attn_config = {
                    "full_attn_layers": full_layers,
                    "kda_layers": [
                        idx
                        for idx in range(fallback.num_hidden_layers)
                        if idx not in set(full_layers)
                    ],
                }
                fallback.first_k_dense_replace = read_field_int(
                    f"{architecture}.leading_dense_block_count", 1
                )
                fallback.num_experts = read_field_int(f"{architecture}.expert_count", 0)
                fallback.num_experts_per_token = read_field_int(
                    f"{architecture}.expert_used_count", 1
                )
                fallback.num_shared_experts = read_field_int(
                    f"{architecture}.expert_shared_count", 0
                )
                fallback.moe_intermediate_size = read_field_int(
                    f"{architecture}.expert_feed_forward_length",
                    fallback.intermediate_size,
                )
                # Conservative defaults to keep dimensions valid for Lite Kimi path.
                fallback.qk_nope_head_dim = 128
                fallback.qk_rope_head_dim = 64
                fallback.v_head_dim = 128
                fallback.kv_lora_rank = read_field_int(
                    f"{architecture}.attention.kv_lora_rank", 512
                )
            else:
                fallback.architectures = ["LlamaForCausalLM"]
            return fallback
            
    # Fallback to transformers for standard models
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)

def benchmark_real_model(name, model_path, batch_size=32, context_len=11, env_overrides=None):
    print(f"\n>>> [REAL] Benchmarking: {name}")
    print(f">>> Path: {model_path}, BS: {batch_size}, Ctx: {context_len}")
    
    clear_gguf_cache()
    gc.collect()
    
    with _temporary_env(env_overrides):
        # 1. Load Real Config
        hf_config = load_real_config(model_path)

        class FakeVllmConfig:
            def __init__(self):
                # Extract attributes safely
                n_kv_heads = getattr(
                    hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 1)
                )
                h_size = getattr(hf_config, "hidden_size", 4096)
                n_heads = getattr(hf_config, "num_attention_heads", 32)
                n_layers = getattr(hf_config, "num_hidden_layers", 32)

                model_dtype = _infer_model_dtype(hf_config)
                self.model_config = type('obj', (object,), {
                    'hf_config': hf_config,
                    'dtype': model_dtype,
                    'max_model_len': 4096,
                    'model': model_path,
                    'get_num_kv_heads': lambda x: n_kv_heads,
                    'get_head_size': lambda: h_size // n_heads,
                    'get_num_layers': lambda x: n_layers,
                    'get_total_num_kv_heads': lambda: n_kv_heads,
                    'get_max_model_len': lambda: 4096,
                })
                self.parallel_config = type('obj', (object,), {
                    'tensor_parallel_size': 1,
                    'pipeline_parallel_size': 1,
                    'world_size': 1,
                })
                # Auto-detect quantization
                self.quant_config = None
                if _is_gguf_dir(model_path):
                    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                    self.quant_config = GGUFConfig()

        v_config = FakeVllmConfig()

        # 2. Load Model & Weights
        print("Loading weights...")
        model = get_model(v_config).cuda().to(dtype=v_config.model_config.dtype)
        _assert_quant_weights_ready(model)

        # 3. Prepare Inputs
        input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
        positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + (context_len - 1)

        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_size = hf_config.hidden_size // num_heads

        kv_caches = []
        num_blocks = 256 # Safe default
        for _ in range(hf_config.num_hidden_layers):
            k_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            v_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            kv_caches.append((k_cache, v_cache))

        attn_metadata = {
            "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
            "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
            "block_tables": torch.zeros((batch_size, num_blocks), device="cuda", dtype=torch.int32)
        }

        # 4. Warmup
        print("Warmup...")
        for _ in range(5):
            with torch.inference_mode():
                model(input_ids, positions, kv_caches, attn_metadata)
        torch.cuda.synchronize()

        # 5. Benchmark
        iters = 20
        start_time = time.time()
        for _ in range(iters):
            with torch.inference_mode():
                model(input_ids, positions, kv_caches, attn_metadata)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_latency = (end_time - start_time) / iters * 1000
        tps = (1000 / avg_latency) * batch_size

        print(f"RESULT: Latency={avg_latency:.2f}ms, Throughput={tps:.2f} tokens/sec")

        del model, kv_caches
        clear_gguf_cache()
        gc.collect()
        return tps

if __name__ == "__main__":
    test_matrix = [
        ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 32, 11, {}),
        ("Qwen3.5-9B-GGUF", "models/Qwen3.5-9B-GGUF", 32, 4096, {"FASTINFERENCE_QWEN9_AGGRESSIVE": "1"}),
        ("GLM-4.7-Flash-GGUF", "models/GLM-4.7-Flash-GGUF", 32, 11, {}),
        ("DeepSeek-V2-Lite-GGUF", "models/DeepSeek-V2-Lite-GGUF", 32, 11, {}),
        ("Kimi-Linear-48B-GGUF", "models/Kimi-Linear-48B-GGUF", 1, 11, {}),
        (
            "Qwen3.5-35B-MoE-GGUF",
            "models/Qwen3.5-35B-MoE-GGUF",
            1,
            1024,
            {"FASTINFERENCE_QWEN35_FINITE_CHECK_INTERVAL": "64"},
        ),
    ]
    
    results = {}
    for name, path, bs, ctx_len, env_overrides in test_matrix:
        if os.path.exists(path):
            try:
                results[name] = benchmark_real_model(
                    name,
                    path,
                    batch_size=bs,
                    context_len=ctx_len,
                    env_overrides=env_overrides,
                )
            except Exception as e:
                print(f"Failed to benchmark {name}: {e}")
                # import traceback
                # traceback.print_exc()
        else:
            print(f"Skipping {name}, path not found: {path}")

    print("\n" + "="*50)
    print("FINAL REAL-WEIGHT PERFORMANCE SUMMARY")
    print("="*50)
    for name, tps in results.items():
        print(f"{name:25}: {tps:8.2f} tokens/sec")
    print("="*50)
