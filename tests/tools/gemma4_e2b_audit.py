#!/usr/bin/env python3
"""P0 audit for Gemma4-E2B decode overhead.

Cold audit: no warmup, captures asymmetric fallback events and dense cache bytes.
Perf run:   warmup, then decode-only TPS and generated token IDs.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.tensor import (
    get_awq_runtime_audit_summary,
    get_awq_runtime_prefix_stats,
    get_awq_runtime_stats,
    reset_awq_runtime_stats,
)


def build_prompt_of_length(tokenizer, text_seed: str, target_len: int) -> list[int]:
    seed_ids = tokenizer.encode(text_seed, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("seed text encoded to empty ids")
    repeats = (target_len // len(seed_ids)) + 2
    long_ids = (seed_ids * repeats)[:target_len]
    assert len(long_ids) == target_len
    return long_ids


def _add_request_with_token_ids(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    engine = llm.engine
    if engine.request_builder is None:
        engine.add_request(
            "__init_request_builder__",
            "",
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        engine.abort_request("__init_request_builder__")
    assert engine.request_builder is not None
    req = engine.request_builder.build(
        request_id=request_id,
        prompt="",
        sampling_params=sampling_params,
    )
    req.input_ids = list(input_ids)
    req.guarded_prompt = ""
    engine.scheduler.enqueue_request(request_id, req)
    admitted = engine.scheduler.admit_queued_requests(max_new=1)
    assert request_id in admitted, f"request {request_id} was not admitted"


def decode_only_generate(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    max_new_tokens: int,
    reset_stats: bool,
) -> dict[str, Any]:
    if reset_stats:
        reset_awq_runtime_stats()

    _add_request_with_token_ids(
        llm,
        request_id,
        input_ids,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True),
    )

    step_budget = max(64, (len(input_ids) + max_new_tokens) * 4)
    final_output: Any | None = None
    first_decode_start: float | None = None
    step_count = 0

    try:
        while llm.engine.active_request_count > 0 and step_count < step_budget:
            step_count += 1
            outs = llm.engine.step()
            for out in outs:
                if out.request_id != request_id:
                    continue
                if (
                    first_decode_start is None
                    and out.outputs
                    and len(out.outputs[0].token_ids) > 0
                ):
                    first_decode_start = time.perf_counter()
                if out.finished:
                    final_output = out
    finally:
        if final_output is None:
            with contextlib.suppress(Exception):
                llm.engine.abort_request(request_id)

    if final_output is None:
        raise RuntimeError(
            f"request {request_id} did not finish within step_budget={step_budget}"
        )

    end = time.perf_counter()
    first_decode_start = first_decode_start or end
    decode_elapsed = end - first_decode_start
    token_ids = list(final_output.outputs[0].token_ids)
    decode_tokens = max(0, len(token_ids) - 1)

    return {
        "decode_time_s": round(decode_elapsed, 3),
        "decode_tps": (
            round(decode_tokens / decode_elapsed, 3) if decode_elapsed > 0 else 0.0
        ),
        "decode_tokens": decode_tokens,
        "token_ids": token_ids,
        "awq_stats": get_awq_runtime_stats(),
    }


def run_cold_audit(llm, prompt_ids, max_new_tokens):
    """Single un-warmed request; do NOT reset stats before or after.
    Captures first-fallback events and dense cache establishment."""
    req_id = f"e2b_cold_{time.time_ns()}"
    result = decode_only_generate(
        llm, req_id, prompt_ids, max_new_tokens, reset_stats=False
    )
    return {
        "decode_tps": result["decode_tps"],
        "token_ids": result["token_ids"],
        "awq_stats": result["awq_stats"],
        "awq_prefix_stats": get_awq_runtime_prefix_stats(),
        "awq_audit_summary": get_awq_runtime_audit_summary(),
    }


def run_perf(llm, prompt_ids, max_new_tokens, repetitions):
    """Warmup once, then measured repetitions with per-run stats reset."""
    warmup_id = f"e2b_warmup_{time.time_ns()}"
    decode_only_generate(llm, warmup_id, prompt_ids, max_new_tokens, reset_stats=True)

    runs = []
    for i in range(repetitions):
        req_id = f"e2b_perf_{i}_{time.time_ns()}"
        runs.append(
            decode_only_generate(
                llm, req_id, prompt_ids, max_new_tokens, reset_stats=True
            )
        )

    runs_sorted = sorted(runs, key=lambda r: r["decode_tps"])
    median = runs_sorted[len(runs_sorted) // 2]
    return {
        "repetitions": repetitions,
        "median_decode_tps": median["decode_tps"],
        "median_decode_time_s": median["decode_time_s"],
        "token_ids": median["token_ids"],
        "awq_stats": median["awq_stats"],
    }


def run_audit(
    model: str,
    text_seed: str,
    context_bucket: int,
    max_new_tokens: int,
    repetitions: int,
    kernel_policy_json: str,
) -> dict[str, Any]:
    policy = json.loads(kernel_policy_json)
    fastinference_config = {
        "tuning_keyvals": {
            "FASTINFERENCE_AWQ_ASYMMETRIC_GEMV": (
                "1" if policy.get("awq_asymmetric_gemv") else "0"
            ),
        }
    }

    llm = LLM(
        model=model,
        max_model_len=context_bucket + max_new_tokens + 16,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        fastinference_config=fastinference_config,
    )
    tokenizer = llm.tokenizer
    prompt_ids = build_prompt_of_length(tokenizer, text_seed, context_bucket)

    return {
        "context_bucket": context_bucket,
        "actual_context_tokens": len(prompt_ids),
        "kernel_policy": policy,
        "cold_audit": run_cold_audit(llm, prompt_ids, max_new_tokens),
        "perf_run": run_perf(llm, prompt_ids, max_new_tokens, repetitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gemma-4-E2B-it-AWQ-INT4")
    parser.add_argument("--context-bucket", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--kernel-policy", default="{}")
    parser.add_argument("--out", default="/tmp/gemma4_e2b_audit.json")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    text_seed = "The capital of France is Paris. "
    result = run_audit(
        args.model,
        text_seed,
        args.context_bucket,
        args.max_new_tokens,
        args.repetitions,
        args.kernel_policy,
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
