# SPDX-License-Identifier: Apache-2.0
"""Cached verifier runner and multi-step microbench for Gemma4 speculative decoding.

Builds prefill-style attention metadata that reuses an existing request's KV
block table, runs the target model with
``attn_metadata["verifier_return_all_logits"] == True``, and returns per-position
logits for the new tokens.

The microbench drives a dual-cache state machine: the draft model generates K
greedy tokens from a shared prefix, the target verifies them in a single forward,
and accepted tokens are committed while rejected tokens trigger a block-table
truncate so both caches stay aligned.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm import LLM
from vllm.kernels.triton.compute_slot_mapping import compute_slot_mapping
from vllm.lora.mapping import LoRAMapping
from vllm.sampling_params import SamplingParams


def build_verifier_metadata(
    kv_block_manager: Any,
    inf_config: Any,
    slot_idx: int,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,
    num_layers: int,
) -> dict[str, Any]:
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    total_len = prefix_len + seqlen

    # Grow the block table if the verifier will write past the current end.
    kv_block_manager.ensure_blocks(request_id, total_len)
    kv_block_manager.update_block_table_row(slot_idx, request_id)

    positions = torch.arange(
        prefix_len, total_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)
    block_table = kv_block_manager.block_table_for_slot(slot_idx).unsqueeze(0)
    query_start_loc = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    slot_mapping = torch.empty(seqlen, device=device, dtype=torch.long)
    compute_slot_mapping(
        query_start_loc,
        positions.view(-1),
        block_table,
        kv_block_manager.block_size,
        slot_mapping,
        pad_id=-1,
    )
    return {
        "slot_mapping": slot_mapping,
        "seq_lens": seq_lens,
        "seq_lens_cpu": [int(total_len)],
        "max_seq_len_cpu": int(total_len),
        "kv_start_indices": torch.tensor(
            [prefix_len], device=device, dtype=torch.int32
        ),
        "kv_start_indices_cpu": [prefix_len],
        "block_tables": block_table,
        "is_prefill": True,
        "verifier_return_all_logits": True,
        "kv_scale_cache": kv_block_manager.kv_scale_caches,
        "kv_cache_dtype": inf_config.kv_type,
        "k_scale": inf_config.k_scale,
        "v_scale": inf_config.v_scale,
        "config": inf_config,
        "lora_mapping": LoRAMapping.from_ids([None]),
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
    }


def run_cached_verifier(
    target_llm: LLM,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,  # (1, K+1)
) -> torch.Tensor:
    engine = target_llm.engine
    kvbm = engine.kv_block_manager
    req = engine.scheduler.get_request(request_id)
    slot_idx = int(req.slot_idx)
    attn_metadata = build_verifier_metadata(
        kvbm,
        engine.inf_config,
        slot_idx,
        request_id,
        prefix_len,
        input_ids,
        len(engine.model.model.layers),
    )
    positions = torch.arange(
        prefix_len,
        prefix_len + input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    with torch.inference_mode():
        logits = engine.model(
            input_ids,
            positions,
            kvbm.kv_caches,
            attn_metadata,
        )
    return logits


@dataclass
class SpecState:
    target_seq_len: int
    draft_seq_len: int
    generated: list[int]
    last_emitted: int
    target_forwards: int = 0
    draft_forwards: int = 0
    catchup_forwards: int = 0


def _add_request_with_token_ids(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    """Enqueue a request using exact token IDs, bypassing text tokenization."""
    engine = llm.engine
    # The request_builder is lazily created on the first public add_request.
    if engine.request_builder is None:
        # Trigger initialization with a prompt that is immediately aborted.
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


def _truncate_request_blocks(
    kv_block_manager: Any,
    request_id: str,
    new_seq_len: int,
) -> None:
    """Free blocks beyond new_seq_len and refresh the block-table row."""
    block_size = kv_block_manager.block_size
    needed_blocks = max(1, (new_seq_len + block_size - 1) // block_size)
    block_ids = kv_block_manager._request_blocks.get(request_id, [])
    if len(block_ids) > needed_blocks:
        tail_ids = block_ids[needed_blocks:]
        block_ids[needed_blocks:] = []
        kv_block_manager._allocator.free(tail_ids)
    slot_idx = kv_block_manager._request_slot_id.get(request_id)
    if slot_idx is not None:
        kv_block_manager.update_block_table_row(slot_idx, request_id)


def run_cached_draft_k(
    draft_llm: LLM,
    request_id: str,
    prefix_len: int,
    first_input: int,
    k: int,
) -> tuple[list[int], int]:
    """Run K greedy decode steps on the draft model using cached KV.

    Returns (draft_token_ids, final_draft_seq_len).
    """
    engine = draft_llm.engine
    kvbm = engine.kv_block_manager
    req = engine.scheduler.get_request(request_id)
    slot_idx = int(req.slot_idx)
    num_layers = len(engine.model.model.layers)
    device = engine.device

    draft_tokens: list[int] = []
    current_input = first_input
    current_seq_len = prefix_len

    for _ in range(k):
        kvbm.ensure_blocks(request_id, current_seq_len + 1)
        kvbm.update_block_table_row(slot_idx, request_id)

        input_ids = torch.tensor([[current_input]], device=device, dtype=torch.long)
        positions = torch.tensor([[current_seq_len]], device=device, dtype=torch.long)
        seq_lens = torch.tensor([current_seq_len + 1], device=device, dtype=torch.int32)
        block_table = kvbm.block_table_for_slot(slot_idx).unsqueeze(0)
        query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
        slot_mapping = torch.empty(1, device=device, dtype=torch.long)
        compute_slot_mapping(
            query_start_loc,
            positions.view(-1),
            block_table,
            kvbm.block_size,
            slot_mapping,
            pad_id=-1,
        )

        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "seq_lens_cpu": [int(current_seq_len + 1)],
            "max_seq_len_cpu": int(current_seq_len + 1),
            "positions_cpu": [int(current_seq_len)],
            "block_tables": block_table,
            "is_prefill": False,
            "kv_start_indices": positions.to(torch.int32),
            "kv_start_indices_cpu": [int(current_seq_len)],
            "linear_attn_carry": [None] * num_layers,
            "linear_conv_carry": [None] * num_layers,
            "kv_scale_cache": kvbm.kv_scale_caches,
            "kv_cache_dtype": engine.inf_config.kv_type,
            "k_scale": engine.inf_config.k_scale,
            "v_scale": engine.inf_config.v_scale,
            "config": engine.inf_config,
            "lora_mapping": LoRAMapping.from_ids([None]),
        }

        with torch.inference_mode():
            logits = engine.model(input_ids, positions, kvbm.kv_caches, attn_metadata)

        next_token = int(torch.argmax(logits[0, -1]).item())
        draft_tokens.append(next_token)
        current_input = next_token
        current_seq_len += 1

        # Keep the scheduler request in sync with the KV cache.
        req.seq_len = current_seq_len
        req.generated_ids.append(next_token)

    return draft_tokens, current_seq_len


def _prefill_target(llm: LLM, prompt_ids: list[int]) -> tuple[str, int]:
    """Prefill target and return (request_id, first_output_token).

    Stops as soon as the first output token appears; the request remains alive
    for subsequent verifier forwards. After prefill, the cached length is reset
    to the prompt length and the first output token is returned as the initial
    last_emitted.
    """
    req_id = f"target_prefill_{time.time_ns()}"
    _add_request_with_token_ids(
        llm,
        req_id,
        prompt_ids,
        SamplingParams(temperature=0.0, max_tokens=999999),
    )
    first_token: int | None = None
    step_budget = max(64, len(prompt_ids) * 2)
    steps = 0
    while first_token is None and steps < step_budget:
        steps += 1
        outs = llm.engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            tokens = list(out.outputs[0].token_ids)
            if tokens:
                first_token = int(tokens[0])
                break
    if first_token is None:
        raise RuntimeError(f"target prefill for {req_id} produced no output token")

    req = llm.engine.scheduler.get_request(req_id)
    req.seq_len = len(prompt_ids)
    req.generated_ids = []
    req.is_prefill = False
    return req_id, first_token


def _prefill_draft_persistent(llm: LLM, prompt_ids: list[int]) -> str:
    """Prefill draft and keep request alive for manual K-step decode."""
    req_id = f"draft_prefill_{time.time_ns()}"
    _add_request_with_token_ids(
        llm,
        req_id,
        prompt_ids,
        SamplingParams(temperature=0.0, max_tokens=999999),
    )
    saw_output = False
    step_budget = max(64, len(prompt_ids) * 2)
    steps = 0
    while not saw_output and steps < step_budget:
        steps += 1
        outs = llm.engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            tokens = list(out.outputs[0].token_ids)
            if tokens:
                saw_output = True
                break

    if not saw_output:
        raise RuntimeError(f"draft prefill for {req_id} produced no output token")

    # Establish the invariant: cached length is exactly the prompt length,
    # and no generated tokens have been consumed yet.
    req = llm.engine.scheduler.get_request(req_id)
    req.seq_len = len(prompt_ids)
    req.generated_ids = []
    req.is_prefill = False
    return req_id


def _get_eos_token_ids(llm: LLM) -> set[int]:
    """Return the EOS token IDs used by the engine baseline."""
    eos_ids: set[int] = set()

    def _add(eos: Any) -> None:
        if eos is None:
            return
        if isinstance(eos, int):
            eos_ids.add(eos)
        else:
            with suppress(Exception):
                eos_ids.update(int(x) for x in eos)

    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is not None:
        _add(getattr(tokenizer, "eos_token_id", None))

    engine = getattr(llm, "engine", None)
    model_config = getattr(engine, "model_config", None) if engine is not None else None
    hf_config = (
        getattr(model_config, "hf_config", None) if model_config is not None else None
    )
    if hf_config is not None:
        _add(getattr(hf_config, "eos_token_id", None))

    return eos_ids


def step_state_machine(
    target_llm: LLM,
    draft_llm: LLM,
    state: SpecState,
    k: int,
    target_req_id: str,
    draft_req_id: str,
) -> None:
    """Execute one speculative decoding step on the dual-cache state."""
    target_engine = target_llm.engine
    draft_engine = draft_llm.engine
    target_kvbm = target_engine.kv_block_manager
    draft_kvbm = draft_engine.kv_block_manager
    target_req = target_engine.scheduler.get_request(target_req_id)
    draft_req = draft_engine.scheduler.get_request(draft_req_id)
    device = target_engine.device

    if k == 0:
        # Terminating tail: only one continuation token remains.
        input_ids = torch.tensor(
            [[state.last_emitted]], device=device, dtype=torch.long
        )
        logits = run_cached_verifier(
            target_llm, target_req_id, state.target_seq_len, input_ids
        )
        state.target_forwards += 1
        next_token = int(torch.argmax(logits[:, 0]).item())
        state.generated.append(next_token)
        state.last_emitted = next_token
        state.target_seq_len += 1
        target_req.seq_len = state.target_seq_len
        target_req.generated_ids = []
        return

    # 1. Draft generates K tokens from [prefix, last_emitted].
    draft_tokens, final_draft_seq_len = run_cached_draft_k(
        draft_llm, draft_req_id, state.draft_seq_len, state.last_emitted, k
    )
    state.draft_forwards += k
    state.draft_seq_len = final_draft_seq_len

    # 2. Verifier runs on [last_emitted, d_1, ..., d_K].
    verifier_input = [state.last_emitted] + draft_tokens
    input_ids = torch.tensor([verifier_input], device=device, dtype=torch.long)
    logits = run_cached_verifier(
        target_llm, target_req_id, state.target_seq_len, input_ids
    )
    state.target_forwards += 1

    # 3. Accept/reject loop produces committed tokens.
    accepted: list[int] = []
    rejected = False
    for i, draft_tok in enumerate(draft_tokens):
        pred = int(torch.argmax(logits[0, i]).item())
        if pred == draft_tok:
            accepted.append(draft_tok)
        else:
            accepted.append(pred)
            rejected = True
            break

    if not rejected:
        # All drafts accepted: take one bonus target token.
        bonus = int(torch.argmax(logits[0, k]).item())
        accepted.append(bonus)

    # 4. Update target_seq_len per the dual-cache commit table and sync request.
    state.target_seq_len += len(accepted)
    target_req.seq_len = state.target_seq_len
    target_req.generated_ids = []

    if rejected:
        # Partial reject: truncate both caches to the accepted prefix length.
        state.draft_seq_len = state.target_seq_len
        _truncate_request_blocks(target_kvbm, target_req_id, state.target_seq_len)
        _truncate_request_blocks(draft_kvbm, draft_req_id, state.draft_seq_len)
        draft_req.seq_len = state.draft_seq_len
        draft_req.generated_ids = []
    else:
        # All accepted: run one draft catch-up forward with d_K, discard output.
        _, final_draft_seq_len = run_cached_draft_k(
            draft_llm, draft_req_id, state.draft_seq_len, draft_tokens[-1], 1
        )
        state.catchup_forwards += 1
        state.draft_seq_len = final_draft_seq_len
        draft_req.seq_len = state.draft_seq_len
        draft_req.generated_ids = []

    # Append committed tokens; the last one becomes the new last_emitted.
    for tok in accepted:
        state.generated.append(tok)
        state.last_emitted = tok


def _baseline_greedy_token_ids(
    llm: LLM, prompt_ids: list[int], max_new_tokens: int
) -> dict[str, Any]:
    """Greedy baseline decode using exact token IDs.

    Returns baseline token IDs and decode TPS. The first output token is treated
    as produced by prefill and is excluded from decode TPS.
    """
    req_id = f"baseline_greedy_{time.time_ns()}"
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, min_tokens=1)
    _add_request_with_token_ids(llm, req_id, prompt_ids, sp)

    final_output: Any | None = None
    first_decode_start: float | None = None
    step_budget = max(64, (len(prompt_ids) + max_new_tokens) * 4)
    steps = 0
    try:
        while llm.engine.active_request_count > 0 and steps < step_budget:
            steps += 1
            outs = llm.engine.step()
            for out in outs:
                if out.request_id != req_id:
                    continue
                if (
                    first_decode_start is None
                    and out.outputs
                    and len(out.outputs[0].token_ids) > 0
                ):
                    first_decode_start = time.perf_counter()
                if out.finished:
                    final_output = out

        if final_output is None:
            raise RuntimeError(
                f"baseline greedy did not finish within step_budget={step_budget}; "
                f"active_request_count={llm.engine.active_request_count}"
            )
    finally:
        if final_output is None:
            with suppress(Exception):
                llm.engine.abort_request(req_id)

    baseline_end = time.perf_counter()
    first_decode_start = first_decode_start or baseline_end

    baseline_token_ids = list(final_output.outputs[0].token_ids)
    decode_tokens = max(0, len(baseline_token_ids) - 1)
    decode_elapsed = baseline_end - first_decode_start
    baseline_decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0

    return {
        "token_ids": baseline_token_ids,
        "decode_tps": baseline_decode_tps,
        "decode_tokens": decode_tokens,
    }


def _load_prompts(prompt_file: str) -> list[dict[str, Any]]:
    """Load the prompt fixture as a list of prompt entries."""
    data = json.loads(Path(prompt_file).read_text())
    return list(data["prompts"])


def _default_target_model() -> str:
    return "models/gemma-4-26B-A4B-it-AWQ-4bit"


def _default_draft_model() -> str:
    return "models/gemma-4-E2B-it-AWQ-INT4"


def _default_prompt_file() -> str:
    return str(
        Path(__file__).resolve().parent / "fixtures" / "gemma4_speculative_prompts.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P1.5 multi-step speculative decoding microbench for Gemma4.",
    )
    parser.add_argument("--target-model", default=_default_target_model())
    parser.add_argument("--draft-model", default=_default_draft_model())
    parser.add_argument("--prompt-file", default=_default_prompt_file())
    parser.add_argument(
        "--k-values",
        type=lambda s: [int(x.strip()) for x in s.split(",") if x.strip()],
        default="1,2,4,8",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--num-repetitions", type=int, default=5)
    parser.add_argument("--tokens-per-repetition", type=int, default=32)
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.isdir(args.target_model):
        print(
            f"[ERROR] target model directory not found: {args.target_model}",
            file=sys.stderr,
        )
        return 2
    if not os.path.isdir(args.draft_model):
        print(
            f"[ERROR] draft model directory not found: {args.draft_model}",
            file=sys.stderr,
        )
        return 2

    prompts = _load_prompts(args.prompt_file)

    max_context = max(p["context_len"] for p in prompts)
    max_new_tokens = max(p["max_new_tokens"] for p in prompts)
    max_k = max(args.k_values)
    needed_model_len = max_context + max_new_tokens + max_k + 32

    target_llm = LLM(
        model=args.target_model,
        max_model_len=needed_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
    )
    draft_llm = LLM(
        model=args.draft_model,
        max_model_len=needed_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
    )

    try:
        eos_ids = _get_eos_token_ids(target_llm)
        tokenizer = getattr(target_llm, "tokenizer", None)

        per_k_results: list[dict[str, Any]] = []
        overall_passed = True

        for k in args.k_values:
            prompt_results: list[dict[str, Any]] = []
            for prompt_item in prompts:
                prompt_text = prompt_item["text"]
                prompt_ids: list[int] = []
                if tokenizer is not None:
                    try:
                        prompt_ids = tokenizer.encode(prompt_text)
                    except Exception:
                        prompt_ids = []

                baseline_result = _baseline_greedy_token_ids(
                    target_llm, prompt_ids, args.tokens_per_repetition + 1
                )
                baseline_token_ids = baseline_result["token_ids"]
                baseline_decode_tps = baseline_result["decode_tps"]

                repetition_results: list[dict[str, Any]] = []
                for _rep in range(args.num_repetitions):
                    target_req_id: str | None = None
                    draft_req_id: str | None = None
                    try:
                        target_req_id, target_y = _prefill_target(
                            target_llm, prompt_ids
                        )
                        draft_req_id = _prefill_draft_persistent(draft_llm, prompt_ids)
                        target_snapshot_len = len(prompt_ids)
                        draft_snapshot_len = len(prompt_ids)

                        state = SpecState(
                            target_seq_len=target_snapshot_len,
                            draft_seq_len=draft_snapshot_len,
                            generated=[],
                            last_emitted=target_y,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        while len(state.generated) < args.tokens_per_repetition:
                            remaining = args.tokens_per_repetition - len(
                                state.generated
                            )
                            effective_k = min(k, remaining - 1) if remaining > 1 else 0
                            step_state_machine(
                                target_llm,
                                draft_llm,
                                state,
                                effective_k,
                                target_req_id,
                                draft_req_id,
                            )
                            if state.last_emitted in eos_ids:
                                break
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                    finally:
                        if target_req_id is not None:
                            target_llm.engine.abort_request(target_req_id)
                        if draft_req_id is not None:
                            draft_llm.engine.abort_request(draft_req_id)

                    actual_committed = len(state.generated)
                    repetition_results.append(
                        {
                            "committed_tokens": actual_committed,
                            "elapsed_s": elapsed,
                            "predicted_tps": (
                                actual_committed / elapsed if elapsed > 0 else 0.0
                            ),
                            "target_forwards": state.target_forwards,
                            "draft_forwards": state.draft_forwards,
                            "catchup_forwards": state.catchup_forwards,
                        }
                    )

                predicted_tps_values = [r["predicted_tps"] for r in repetition_results]
                aggregate_predicted_tps = statistics.median(predicted_tps_values)
                speedup = (
                    aggregate_predicted_tps / baseline_decode_tps
                    if baseline_decode_tps > 0
                    else 0.0
                )

                actual_committed = len(state.generated)
                bit_exact = (
                    state.generated == baseline_token_ids[1 : 1 + actual_committed]
                )

                per_prompt_passed = (
                    bit_exact
                    and speedup >= 1.2
                    and aggregate_predicted_tps >= 13.0
                    and aggregate_predicted_tps >= baseline_decode_tps
                )

                prompt_results.append(
                    {
                        "id": prompt_item["id"],
                        "context_len": prompt_item["context_len"],
                        "max_new_tokens": prompt_item["max_new_tokens"],
                        "bit_exact": bit_exact,
                        "baseline_decode_tps": baseline_decode_tps,
                        "aggregate_predicted_tps": aggregate_predicted_tps,
                        "speedup": speedup,
                        "passed": per_prompt_passed,
                        "repetition_results": repetition_results,
                        "baseline_token_ids": baseline_token_ids,
                        "generated_token_ids": list(state.generated),
                    }
                )

            aggregate_predicted_tps = statistics.median(
                [r["aggregate_predicted_tps"] for r in prompt_results]
            )
            aggregate_baseline_tps = statistics.median(
                [r["baseline_decode_tps"] for r in prompt_results]
            )
            aggregate_speedup = (
                aggregate_predicted_tps / aggregate_baseline_tps
                if aggregate_baseline_tps > 0
                else 0.0
            )
            k_passed = (
                all(r["bit_exact"] for r in prompt_results)
                and aggregate_speedup >= 1.2
                and aggregate_predicted_tps >= 13.0
                and all(
                    r["aggregate_predicted_tps"] >= r["baseline_decode_tps"]
                    for r in prompt_results
                )
            )

            if not k_passed:
                overall_passed = False

            per_k_results.append(
                {
                    "k": k,
                    "aggregate_predicted_tps": aggregate_predicted_tps,
                    "aggregate_baseline_tps": aggregate_baseline_tps,
                    "aggregate_speedup": aggregate_speedup,
                    "passed": k_passed,
                    "prompts": prompt_results,
                }
            )

        output = {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "k_values": args.k_values,
            "num_repetitions": args.num_repetitions,
            "tokens_per_repetition": args.tokens_per_repetition,
            "overall_passed": overall_passed,
            "per_k": per_k_results,
        }

        print(json.dumps(output, indent=2, sort_keys=True))
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n"
            )

        return 0 if overall_passed else 1
    finally:
        target_llm.shutdown()
        draft_llm.shutdown()


if __name__ == "__main__":
    sys.exit(main())
