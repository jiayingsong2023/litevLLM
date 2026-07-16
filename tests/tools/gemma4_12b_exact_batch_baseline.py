# SPDX-License-Identifier: Apache-2.0
"""Measure Gemma4 12B exact batch-decode envelopes in isolated processes.

Run one process per ``--batch-size``.  AWQ linear weights cache their M=1
dispatch decision, so combining M=1 and M>1 in one process would make the
fallback audit ambiguous.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "models/gemma-4-12B-it-AWQ-INT4"
DEFAULT_FIXTURES = Path(__file__).parent / "fixtures/gemma4_12b_batch_prompts.json"
_KV_BLOCK_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument(
        "--surface",
        choices=("engine", "llm"),
        default="engine",
        help="engine measures decode-only TPS; llm exercises LLM.generate end to end.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--context-tokens",
        type=int,
        choices=(128, 512, 2048),
        help="Inject exact token-ID prompts for a paged-attention parity bucket.",
    )
    parser.add_argument(
        "--paged-attention-debug",
        action="store_true",
        help="Synchronize and validate paged-KV bounds around every attention launch.",
    )
    parser.add_argument(
        "--paged-attn-local",
        nargs=2,
        type=int,
        metavar=("WARPS", "STAGES"),
        help="Tool-only local paged-attention launch override.",
    )
    parser.add_argument(
        "--paged-attn-global",
        nargs=2,
        type=int,
        metavar=("WARPS", "STAGES"),
        help="Tool-only global paged-attention launch override.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--kv-type", choices=("fp16", "fp8"), default="fp8")
    parser.add_argument(
        "--local-decode-triton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit switch for the Gemma4 local decode attention kernel.",
    )
    parser.add_argument(
        "--mlp-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic switch for the existing M=1 AWQ MLP streaming kernel.",
    )
    parser.add_argument(
        "--fused-gate-up-group32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the model policy for the group-oriented M=1 AWQ gate/up kernel.",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--min-speedup",
        type=float,
        help=(
            "Require this aggregate TPS speedup over --reference. "
            "The reference must be an M=1 result from this tool."
        ),
    )
    parser.add_argument(
        "--max-per-agent-p95-slowdown",
        type=float,
        default=2.0,
        help=(
            "Maximum allowed p95 decode inter-token latency ratio against "
            "the M=1 reference; set to 0 to report without gating."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
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


def load_fixtures(path: Path) -> list[dict[str, str]]:
    fixtures = json.loads(path.read_text())
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError(f"fixture file must contain a non-empty list: {path}")
    for fixture in fixtures:
        if not isinstance(fixture.get("id"), str) or not isinstance(
            fixture.get("prompt"), str
        ):
            raise ValueError(f"invalid fixture: {fixture!r}")
    return fixtures


def run_engine_batch(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
    prompt_token_ids: list[list[int]] | None = None,
) -> tuple[list[list[int]], float, list[float | None]]:
    """Run a batch and report token IDs plus decode-only wall-clock TPS."""
    import torch

    request_ids = [
        f"gemma4_12b_m{len(prompts)}_{index}_{time.time_ns()}"
        for index in range(len(prompts))
    ]
    if prompt_token_ids is None:
        for request_id, prompt in zip(request_ids, prompts):
            llm.engine.add_request(request_id, prompt, copy.deepcopy(sampling_params))
    else:
        if llm.engine.request_builder is None:
            bootstrap_id = f"gemma4_12b_bootstrap_{time.time_ns()}"
            llm.engine.add_request(
                bootstrap_id, prompts[0], copy.deepcopy(sampling_params)
            )
            llm.engine.abort_request(bootstrap_id)
        for request_id, prompt, token_ids in zip(
            request_ids, prompts, prompt_token_ids
        ):
            request = llm.engine.request_builder.build(
                request_id=request_id,
                prompt=prompt,
                sampling_params=copy.deepcopy(sampling_params),
            )
            request.input_ids = list(token_ids)
            llm.engine.scheduler.enqueue_request(request_id, request)

    latest: dict[str, Any] = {}
    first_token_seen: set[str] = set()
    last_token_count = {request_id: 0 for request_id in request_ids}
    last_token_event: dict[str, Any] = {}
    decode_interval_events: dict[str, list[tuple[Any, Any, int]]] = {
        request_id: [] for request_id in request_ids
    }
    decode_started = False
    decode_start = 0.0
    try:
        while len(latest) < len(request_ids) or any(
            not latest[request_id].finished
            for request_id in request_ids
            if request_id in latest
        ):
            step_outputs = llm.engine.step()
            step_event = torch.cuda.Event(enable_timing=True)
            step_event.record()
            for output in step_outputs:
                if output.request_id in request_ids:
                    latest[output.request_id] = output
                    token_count = (
                        len(output.outputs[0].token_ids) if output.outputs else 0
                    )
                    if token_count:
                        first_token_seen.add(output.request_id)
                        previous_count = last_token_count[output.request_id]
                        if previous_count > 0 and token_count > previous_count:
                            decode_interval_events[output.request_id].append(
                                (
                                    last_token_event[output.request_id],
                                    step_event,
                                    token_count - previous_count,
                                )
                            )
                        last_token_count[output.request_id] = token_count
                        last_token_event[output.request_id] = step_event
            if not decode_started and len(first_token_seen) == len(request_ids):
                torch.cuda.synchronize()
                decode_start = time.perf_counter()
                decode_started = True
                layers = getattr(getattr(llm.model, "model", None), "layers", ())
                if layers and getattr(
                    layers[0]._layer_config, "profile_enabled", False
                ):
                    layers[0]._layer_config.profile_stats.clear()

        if not decode_started:
            raise RuntimeError("batch completed before producing a decode token")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - decode_start
        token_ids = [
            list(latest[request_id].outputs[0].token_ids) for request_id in request_ids
        ]
        decode_tokens = sum(max(0, len(ids) - 1) for ids in token_ids)
        p95_by_request: list[float | None] = []
        for request_id in request_ids:
            intervals = sorted(
                start.elapsed_time(end) / generated
                for start, end, generated in decode_interval_events[request_id]
            )
            if not intervals:
                p95_by_request.append(None)
                continue
            index = min(len(intervals) - 1, int(len(intervals) * 0.95))
            p95_by_request.append(float(intervals[index]))
        return (
            token_ids,
            decode_tokens / elapsed if elapsed > 0 else 0.0,
            p95_by_request,
        )
    finally:
        for request_id in request_ids:
            with suppress(Exception):
                llm.engine.abort_request(request_id)


def run_llm_batch(
    llm: Any, prompts: list[str], sampling_params: Any
) -> tuple[list[list[int]], float, list[float | None]]:
    """Exercise the public offline API and report end-to-end output TPS."""
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, copy.deepcopy(sampling_params))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    generated_tokens = sum(len(ids) for ids in token_ids)
    return (
        token_ids,
        generated_tokens / elapsed if elapsed > 0 else 0.0,
        [None for _ in token_ids],
    )


def reference_mismatches(
    rows: list[dict[str, Any]], reference: dict[str, Any]
) -> list[str]:
    expected = {row["fixture_id"]: row["token_ids"] for row in reference["fixtures"]}
    actual = {row["fixture_id"]: row["token_ids"] for row in rows}
    return [
        fixture_id
        for fixture_id in actual
        if actual.get(fixture_id) != expected.get(fixture_id)
    ]


def _expand_to_context_bucket(
    token_ids: list[int], filler_ids: list[int], context_tokens: int
) -> list[int]:
    if len(token_ids) >= context_tokens:
        return token_ids[:context_tokens]
    if not filler_ids:
        raise ValueError("context fixture filler must encode to at least one token")
    out = list(token_ids)
    while len(out) < context_tokens:
        out.extend(filler_ids)
    return out[:context_tokens]


def _execution_max_model_len(
    *, max_model_len: int, context_tokens: int | None, max_new_tokens: int
) -> int:
    """Reserve decode headroom and round up to the paged-KV block boundary."""
    required = (context_tokens or 0) + max_new_tokens
    requested = max(int(max_model_len), required)
    return ((requested + _KV_BLOCK_SIZE - 1) // _KV_BLOCK_SIZE) * _KV_BLOCK_SIZE


def main() -> None:
    args = parse_args()
    if args.min_speedup is not None and args.min_speedup <= 0:
        raise ValueError("--min-speedup must be positive")
    if args.min_speedup is not None and args.reference is None:
        raise ValueError("--min-speedup requires --reference")
    if args.batch_size > 1 and args.reference is None:
        raise ValueError("M>1 verification requires an M=1 --reference result")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is not None and int(reference.get("batch_size", 0)) != 1:
        raise ValueError("--reference must be produced with --batch-size 1")
    # Runtime configuration is read during vllm import/engine creation.
    os.environ["FASTINFERENCE_KV_TYPE"] = args.kv_type
    os.environ.pop("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", None)

    import torch

    from vllm import LLM
    from vllm.model_executor.layers.quantization.tensor import (
        get_awq_runtime_audit_summary,
        get_awq_runtime_prefix_stats,
        get_awq_runtime_stats,
        reset_awq_runtime_stats,
    )
    from vllm.sampling_params import SamplingParams

    fixtures = load_fixtures(args.fixtures)
    effective_max_model_len = _execution_max_model_len(
        max_model_len=args.max_model_len,
        context_tokens=args.context_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    llm = LLM(
        args.model,
        max_num_seqs=args.batch_size,
        max_model_len=effective_max_model_len,
        max_num_batched_tokens=effective_max_model_len * args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.context_tokens is not None:
        required_kv_tokens = args.context_tokens + args.max_new_tokens
        actual_kv_tokens = int(llm.engine.max_model_len)
        if actual_kv_tokens < required_kv_tokens:
            raise RuntimeError(
                "paged-attention fixture exceeds the allocated KV capacity: "
                f"required={required_kv_tokens}, allocated={actual_kv_tokens}"
            )
    llm.engine.inf_config.model_policy["local_decode_triton"] = args.local_decode_triton
    llm.engine.inf_config.kernel_policy["awq_mlp_streaming_fusion"] = args.mlp_streaming
    if args.fused_gate_up_group32 is not None:
        llm.engine.inf_config.kernel_policy["awq_fused_gate_up_group32"] = (
            args.fused_gate_up_group32
        )
    if args.paged_attention_debug:
        llm.engine.inf_config.kernel_policy["paged_attention_debug"] = True
        llm.engine.inf_config.kernel_policy["paged_attention_debug_decode_only"] = True
    if args.paged_attn_local is not None:
        (
            llm.engine.inf_config.paged_attn_num_warps_local,
            llm.engine.inf_config.paged_attn_num_stages_local,
        ) = args.paged_attn_local
    if args.paged_attn_global is not None:
        (
            llm.engine.inf_config.paged_attn_num_warps_global,
            llm.engine.inf_config.paged_attn_num_stages_global,
        ) = args.paged_attn_global
    if args.batch_size > 1:
        # The production envelope remains M=1 until every context bucket has
        # passed. This standalone verifier intentionally exercises the
        # candidate M so it can establish or reject that promotion.
        llm.engine.step_scheduler.set_verified_decode_batch_sizes((1, args.batch_size))
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        ignore_eos=True,
    )
    wrapped_prompts = [
        apply_chat_template(llm.tokenizer, item["prompt"]) for item in fixtures
    ]
    fixture_token_ids: list[list[int]] | None = None
    if args.context_tokens is not None:
        if args.surface != "engine":
            raise ValueError("--context-tokens is supported only by --surface engine")
        filler_ids = list(llm.tokenizer.encode(" context parity filler"))
        fixture_token_ids = [
            _expand_to_context_bucket(
                list(llm.tokenizer.encode(prompt)), filler_ids, args.context_tokens
            )
            for prompt in wrapped_prompts
        ]
        assert all(len(ids) == args.context_tokens for ids in fixture_token_ids)

    if len(fixtures) % args.batch_size:
        raise ValueError(
            "fixture count must be divisible by --batch-size to avoid tail batches: "
            f"fixtures={len(fixtures)}, batch_size={args.batch_size}"
        )

    run_batch = run_engine_batch if args.surface == "engine" else run_llm_batch
    throughput_kind = (
        "decode_only_tps" if args.surface == "engine" else "end_to_end_output_tps"
    )
    # Warm the exact batch shape. It intentionally remains outside audit and timing.
    warm_token_ids = (
        fixture_token_ids[: args.batch_size] if fixture_token_ids is not None else None
    )
    if args.surface == "engine":
        run_engine_batch(
            llm,
            wrapped_prompts[: args.batch_size],
            sampling_params,
            warm_token_ids,
        )
    else:
        run_batch(llm, wrapped_prompts[: args.batch_size], sampling_params)
    reset_awq_runtime_stats()

    rows_by_id: dict[str, dict[str, Any]] = {}
    total_tps: list[float] = []
    for batch_start in range(0, len(fixtures), args.batch_size):
        batch_fixtures = fixtures[batch_start : batch_start + args.batch_size]
        batch_prompts = wrapped_prompts[batch_start : batch_start + args.batch_size]
        batch_token_ids = (
            fixture_token_ids[batch_start : batch_start + args.batch_size]
            if fixture_token_ids is not None
            else None
        )
        if args.surface == "engine":
            replicas, throughput_tps, p95_by_request = run_engine_batch(
                llm,
                batch_prompts,
                sampling_params,
                batch_token_ids,
            )
        else:
            replicas, throughput_tps, p95_by_request = run_batch(
                llm, batch_prompts, sampling_params
            )
        for index, (fixture, prompt, token_ids, p95_decode_ms) in enumerate(
            zip(batch_fixtures, batch_prompts, replicas, p95_by_request)
        ):
            rows_by_id[fixture["id"]] = {
                "fixture_id": fixture["id"],
                "prompt_token_count": (
                    len(batch_token_ids[index])
                    if batch_token_ids is not None
                    else len(llm.tokenizer.encode(prompt))
                ),
                "token_ids": token_ids,
                "throughput_tps": throughput_tps,
                "p95_decode_ms": p95_decode_ms,
            }
        total_tps.append(throughput_tps)

    rows = [rows_by_id[fixture["id"]] for fixture in fixtures]

    chat_template = getattr(llm.tokenizer, "chat_template", "") or ""
    chat_template_sha256 = hashlib.sha256(chat_template.encode()).hexdigest()
    reference_contract_mismatches: list[str] = []
    mismatches: list[str] = []
    reference_tps: float | None = None
    if reference is not None:
        if str(reference.get("model")) != args.model:
            reference_contract_mismatches.append("model")
        if str(reference.get("kv_type")) != args.kv_type:
            reference_contract_mismatches.append("kv_type")
        if str(reference.get("chat_template_sha256")) != chat_template_sha256:
            reference_contract_mismatches.append("chat_template_sha256")
        if str(reference.get("execution_surface", "engine")) != args.surface:
            reference_contract_mismatches.append("execution_surface")
        if reference.get("context_tokens") != args.context_tokens:
            reference_contract_mismatches.append("context_tokens")
        mismatches = reference_mismatches(rows, reference)
        reference_tps_value = reference.get("median_throughput_tps")
        if reference_tps_value is None:
            reference_tps_value = reference["median_decode_tps"]
        reference_tps = float(reference_tps_value)
    median_throughput_tps = sorted(total_tps)[len(total_tps) // 2]
    speedup = (
        median_throughput_tps / reference_tps
        if reference_tps is not None and reference_tps > 0
        else None
    )
    meets_speedup_gate = args.min_speedup is None or (
        speedup is not None and speedup >= args.min_speedup
    )
    p95_slowdown_by_fixture: dict[str, float] = {}
    if reference is not None and args.max_per_agent_p95_slowdown > 0:
        expected_rows = {
            str(row["fixture_id"]): row for row in reference.get("fixtures", [])
        }
        for row in rows:
            actual_p95 = row.get("p95_decode_ms")
            expected_p95 = expected_rows.get(str(row["fixture_id"]), {}).get(
                "p95_decode_ms"
            )
            if actual_p95 is None or expected_p95 is None:
                continue
            expected_value = float(expected_p95)
            if expected_value > 0:
                p95_slowdown_by_fixture[str(row["fixture_id"])] = (
                    float(actual_p95) / expected_value
                )
    max_per_agent_p95_slowdown = max(p95_slowdown_by_fixture.values(), default=None)
    meets_p95_gate = (
        args.max_per_agent_p95_slowdown <= 0
        or reference is None
        or (
            max_per_agent_p95_slowdown is not None
            and max_per_agent_p95_slowdown <= args.max_per_agent_p95_slowdown
        )
    )
    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "execution_surface": args.surface,
        "throughput_kind": throughput_kind,
        "kv_type": args.kv_type,
        "local_decode_triton": args.local_decode_triton,
        "mlp_streaming": args.mlp_streaming,
        "fused_gate_up_group32": bool(
            llm.engine.inf_config.kernel_policy.get("awq_fused_gate_up_group32", False)
        ),
        "paged_attn_local": args.paged_attn_local,
        "paged_attn_global": args.paged_attn_global,
        "gpu": torch.cuda.get_device_name(0),
        "chat_template_sha256": chat_template_sha256,
        "fixture_path": str(args.fixtures),
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "reference_bit_exact": not mismatches and not reference_contract_mismatches,
        "reference_mismatches": mismatches,
        "reference_contract_mismatches": reference_contract_mismatches,
        "reference_median_decode_tps": reference_tps,
        "median_throughput_tps": median_throughput_tps,
        "speedup_vs_reference": speedup,
        "min_speedup": args.min_speedup,
        "meets_speedup_gate": meets_speedup_gate,
        "per_agent_p95_decode_ms": {
            str(row["fixture_id"]): row["p95_decode_ms"] for row in rows
        },
        "max_per_agent_p95_slowdown": max_per_agent_p95_slowdown,
        "max_per_agent_p95_slowdown_limit": args.max_per_agent_p95_slowdown,
        "meets_per_agent_p95_gate": meets_p95_gate,
        "verification_passed": (
            not mismatches
            and not reference_contract_mismatches
            and meets_speedup_gate
            and meets_p95_gate
        ),
        "fixtures": rows,
        "awq_runtime_stats": get_awq_runtime_stats(),
        "awq_prefix_stats": get_awq_runtime_prefix_stats(),
        "awq_audit_summary": get_awq_runtime_audit_summary(),
    }
    layers = getattr(getattr(llm.model, "model", None), "layers", ())
    if layers:
        result["gemma4_profile_stats"] = dict(
            getattr(layers[0]._layer_config, "profile_stats", {})
        )
    if args.surface == "engine":
        result["median_decode_tps"] = median_throughput_tps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if reference_contract_mismatches:
        raise RuntimeError(
            f"reference execution contract differs: {reference_contract_mismatches}"
        )
    if mismatches:
        raise RuntimeError(f"batch token IDs differ from M=1 reference: {mismatches}")
    if not meets_speedup_gate:
        raise RuntimeError(
            "batch speedup gate failed: "
            f"measured={speedup:.3f}x required={args.min_speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
