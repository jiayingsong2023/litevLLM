import argparse
import json
import os
import time

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.model_loader import get_tokenizer
from vllm.sampling_params import SamplingParams


DEFAULT_MODEL_PATH = "models/Qwen3.5-35B-AWQ"
DEFAULT_PROMPT = "法国的首都是哪里？"


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    group_size, bits = 128, 4
    config_path = os.path.join(model_path, "config.json")
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        quantization_config = raw_config.get("quantization_config") or {}
        config_groups = quantization_config.get("config_groups")
        if isinstance(config_groups, dict):
            for group_value in config_groups.values():
                if not isinstance(group_value, dict):
                    continue
                weights_config = group_value.get("weights")
                if isinstance(weights_config, dict):
                    if weights_config.get("group_size") is not None:
                        group_size = int(weights_config["group_size"])
                    if weights_config.get("num_bits") is not None:
                        bits = int(weights_config["num_bits"])
                    break
        if quantization_config.get("group_size") is not None:
            group_size = int(quantization_config["group_size"])
        if quantization_config.get("bits") is not None:
            bits = int(quantization_config["bits"])
    except Exception as exc:
        print(f"[Warning] Could not parse AWQ config: {exc}")
    return group_size, bits


def _apply_stable_qwen35_35b_awq_env_defaults() -> None:
    # Compatibility hint for gfx1151-based AI Max+ systems.
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    # Stable FP8-only runtime defaults for Qwen3.5-35B-AWQ on shared-memory AMD systems.
    os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")
    os.environ.setdefault("FASTINFERENCE_QWEN35_MOE_FP8", "1")
    os.environ.setdefault("FASTINFERENCE_QWEN35_MOE_OFFLOAD", "1")
    os.environ.setdefault("FASTINFERENCE_AWQ_FP8", "1")
    os.environ.setdefault("FASTINFERENCE_AWQ_BLOCK_FP8", "1")
    os.environ.setdefault("FASTINFERENCE_QWEN35_PROMPT_GUARD", "1")


def _looks_like_preformatted_chat(text: str) -> bool:
    stripped = text.lstrip()
    if len(stripped) >= 12 and "<|im_start|>" in stripped[:400]:
        return True
    if stripped.startswith("<|") and "user" in stripped[:120].lower():
        return True
    if stripped.startswith("<think>") or "<think>" in stripped[:240]:
        return True
    return False


def _apply_chat_template_if_available(prompt: str, tokenizer) -> str:
    if _looks_like_preformatted_chat(prompt):
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        print(f"[Warning] Chat template failed, using raw prompt: {exc}")
        return prompt


def _build_engine(model_path: str, max_model_len: int, max_num_seqs: int) -> LiteEngine:
    group_size, weight_bits = _read_awq_group_size_and_bits(model_path)
    model_config = ModelConfig(model=model_path, tokenizer=model_path)
    cache_config = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=min(8192, max(512, max_model_len * 4)),
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    load_config = LoadConfig()
    quant_config = AWQConfig(weight_bits=weight_bits, group_size=group_size)
    vllm_config = VllmConfig(
        model_config,
        cache_config,
        scheduler_config,
        load_config,
        quant_config=quant_config,
    )
    engine = LiteEngine(vllm_config)
    engine.tokenizer = get_tokenizer(model_config, trust_remote_code=True)
    return engine


def _build_default_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.24,
        frequency_penalty=0.08,
        presence_penalty=0.08,
        max_tokens=max_tokens,
    )


def _run_single_request(engine: LiteEngine, prompt: str, sampling_params: SamplingParams) -> str:
    request_id = f"stable-qwen35-awq-{int(time.time() * 1000)}"
    wrapped_prompt = _apply_chat_template_if_available(prompt, engine.tokenizer)
    engine.add_request(request_id, wrapped_prompt, sampling_params)
    step_budget = max(256, len(engine.tokenizer.encode(wrapped_prompt)) + int(sampling_params.max_tokens or 16) + 64)
    latest_text = ""
    for _ in range(step_budget):
        for output in engine.step():
            if output.request_id != request_id:
                continue
            latest_text = output.outputs[0].text
            if output.finished:
                return latest_text
        if engine.active_request_count == 0:
            break
    raise RuntimeError(f"Request did not finish within step budget. Latest text: {latest_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stable default entrypoint for Qwen3.5-35B-AWQ on FP8-only AMD shared-memory systems."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Single prompt to run.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max generated tokens.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="KV cache context length.")
    parser.add_argument("--max-num-seqs", type=int, default=4, help="Max active request slots.")
    args = parser.parse_args()

    print(f"Testing stable Qwen3.5-35B-AWQ entrypoint from: {args.model}")
    try:
        _apply_stable_qwen35_35b_awq_env_defaults()
        engine = _build_engine(args.model, args.max_model_len, args.max_num_seqs)
        sampling_params = _build_default_sampling_params(args.max_new_tokens)
        generated_text = _run_single_request(engine, args.prompt, sampling_params)
        print(f"Prompt: {args.prompt!r}")
        print(f"Generated text: {generated_text!r}")
    except Exception as exc:
        print(f"Error during loading/inference: {exc}")


if __name__ == "__main__":
    main()
