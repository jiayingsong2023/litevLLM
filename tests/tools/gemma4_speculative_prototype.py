# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from tests.tools.gemma4_speculative_tokenizer_gate import (
    _load_tokenizer,
    build_report,
)
from vllm import LLM
from vllm.sampling_params import SamplingParams

DEFAULT_PROMPT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "gemma4_speculative_prompts.json"
)


def propose_ngram(
    prefix_token_ids: list[int],
    generated_token_ids: list[int],
    k: int,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> list[int]:
    """Propose up to k draft tokens by matching the recent generated suffix.

    Searches prefix_token_ids + generated_token_ids for the most recent earlier
    occurrence of the suffix of generated_token_ids.  Returns the tokens that
    followed that earlier occurrence, up to k.
    """
    full = prefix_token_ids + generated_token_ids
    for n in range(ngram_max, ngram_min - 1, -1):
        if len(generated_token_ids) < n:
            continue
        needle = tuple(generated_token_ids[-n:])
        # The final occurrence (the suffix itself) ends at len(full) - n.
        # Search before that so we do not match the needle against itself.
        for i in range(len(full) - n - 1, -1, -1):
            if tuple(full[i : i + n]) == needle:
                start = i + n
                return full[start : start + k]
    return []


def run_target_logits(llm, input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-position target logits for greedy speculative verification.

    This bypasses LLM.generate() and runs the inner model on the full sequence
    with no KV cache.  For greedy decoding this is mathematically equivalent to
    the engine's decode path, just slower because attention is recomputed from
    scratch each call.
    """
    inner = llm.model.model
    device = next(inner.parameters()).device

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    kv_caches = [None] * len(inner.layers)
    attn_metadata = {
        "config": llm.engine.inf_config,
        "is_prefill": True,
        "positions_cpu": list(range(seq_len)),
        "max_seq_len_cpu": seq_len,
    }

    with torch.inference_mode():
        hidden = inner(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping=None,
        )

        if getattr(inner.config, "tie_word_embeddings", False):
            logits = torch.nn.functional.linear(hidden, inner.embed_tokens.weight)
        else:
            logits = llm.model.lm_head(hidden, lora_mapping=None)

        final_softcap = getattr(inner.config, "final_logit_softcapping", None)
        if final_softcap is not None and float(final_softcap) > 0:
            logits = torch.tanh(logits / float(final_softcap)) * float(final_softcap)

    return logits


def _get_eos_token_ids(llm: LLM) -> set[int]:
    """Return the EOS token IDs used by the engine baseline.

    Uses exactly the same sources the engine's greedy baseline uses:
    ``llm.tokenizer.eos_token_id`` and
    ``llm.engine.model_config.hf_config.eos_token_id``.
    ``generation_config.json`` is intentionally *not* loaded, because it can
    declare extra EOS ids (e.g. Gemma4 26B's ``<|tool_response|>``) that are not
    part of the engine baseline's EOS set and would cause the speculative path
    to stop early, breaking ``bit_exact``.

    If neither source is available, returns an empty set so generation is only
    bounded by ``max_new_tokens``.
    """
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

    # The engine's greedy baseline honors eos_token_id from the HF config.
    engine = getattr(llm, "engine", None)
    model_config = getattr(engine, "model_config", None) if engine is not None else None
    hf_config = (
        getattr(model_config, "hf_config", None) if model_config is not None else None
    )
    if hf_config is not None:
        _add(getattr(hf_config, "eos_token_id", None))

    return eos_ids


def _looks_like_preformatted_chat(text: str) -> bool:
    s = text.lstrip()
    if len(s) >= 12 and "<|im_start|>" in s[:400]:
        return True
    return s.startswith("<|") and "user" in s[:120].lower()


def _apply_chat_template(tokenizer: Any, text: str) -> str:
    """Wrap a raw user prompt in the model's chat template when one exists."""
    if _looks_like_preformatted_chat(text):
        return text
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return text
    try:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        return text


def baseline_greedy(llm: LLM, prompt_text: str, max_new_tokens: int) -> dict[str, Any]:
    """Greedy baseline run.

    Returns a dict with prompt token IDs, generated token IDs, and split
    prefill/decode timings.  Uses the engine's step loop directly (rather than
    ``LLM.generate``) so that Gemma4's prefill-only first step, which produces
    no outputs, does not abort the run.
    """
    tokenizer = getattr(llm, "tokenizer", None)
    wrapped_prompt = _apply_chat_template(tokenizer, prompt_text)
    sp = SamplingParams(
        temperature=0.0, max_tokens=max_new_tokens, min_tokens=1, top_p=1.0
    )

    req_id = f"baseline_greedy_{time.time_ns()}"
    llm.engine.add_request(req_id, wrapped_prompt, sp)

    prompt_tokens: list[int] = []
    if tokenizer is not None:
        try:
            prompt_tokens = tokenizer.encode(wrapped_prompt)
        except Exception:
            prompt_tokens = []
    step_budget = max(64, (len(prompt_tokens) + max_new_tokens) * 4)

    final_output: Any | None = None
    step_count = 0
    first_decode_start: float | None = None
    baseline_start = time.perf_counter()
    try:
        while llm.engine.active_request_count > 0 and step_count < step_budget:
            step_count += 1
            outs = llm.engine.step()
            for out in outs:
                if out.request_id != req_id:
                    continue
                if first_decode_start is None and len(out.outputs[0].token_ids) > 0:
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

    prefill_elapsed = first_decode_start - baseline_start
    decode_elapsed = baseline_end - first_decode_start
    baseline_token_ids = list(final_output.outputs[0].token_ids)
    # The first token is produced by the prefill step and is not part of decode TPS.
    decode_tokens = max(0, len(baseline_token_ids) - 1)
    baseline_decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0

    return {
        "prompt_token_ids": list(final_output.prompt_token_ids),
        "token_ids": baseline_token_ids,
        "prefill_time_s": prefill_elapsed,
        "decode_time_s": decode_elapsed,
        "decode_tps": baseline_decode_tps,
    }


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
    assert request_id in admitted, f"draft request {request_id} was not admitted"


def generate_draft_tokens_from_ids(
    draft_llm: LLM, context_ids: list[int], k: int
) -> list[int]:
    """Greedy-decode up to k draft tokens from the draft model using token IDs."""
    sp = SamplingParams(temperature=0.0, max_tokens=k, min_tokens=1)
    req_id = f"draft_generate_{time.time_ns()}"
    _add_request_with_token_ids(draft_llm, req_id, context_ids, sp)

    final_output: Any | None = None
    step_count = 0
    step_budget = max(64, k * 4)
    try:
        while draft_llm.engine.active_request_count > 0 and step_count < step_budget:
            step_count += 1
            outs = draft_llm.engine.step()
            for out in outs:
                if out.request_id != req_id:
                    continue
                if out.finished or (
                    final_output is None and len(out.outputs[0].token_ids) >= k
                ):
                    final_output = out

        if final_output is None:
            raise RuntimeError(
                f"draft generation did not finish within step_budget={step_budget}; "
                f"active_request_count={draft_llm.engine.active_request_count}"
            )
    finally:
        if final_output is None:
            with suppress(Exception):
                draft_llm.engine.abort_request(req_id)

    generated = list(final_output.outputs[0].token_ids)
    # Sanity check: the engine consumed exactly the token IDs we provided.
    assert list(final_output.prompt_token_ids) == context_ids, (
        "draft engine prompt token IDs do not match input token IDs"
    )
    return generated[:k]


def speculative_decode(
    llm: LLM,
    draft_proposer: Callable[[list[int], list[int], int], list[int]],
    prompt_token_ids: list[int],
    max_new_tokens: int,
    num_draft_tokens: int,
) -> dict[str, Any]:
    """Offline speculative decode loop using n-gram (or any) draft tokens.

    Verifies draft tokens with run_target_logits and accepts them while they
    match the greedy target.  On the first mismatch the recovered target token
    is used instead.  If all draft tokens are accepted, one bonus target token
    is sampled from the logit past the last accepted draft.

    ``baseline_token_ids`` and ``bit_exact`` are caller-filled placeholders;
    the offline CLI populates them after the baseline run.
    """
    eos_ids = _get_eos_token_ids(llm)
    prefix = list(prompt_token_ids)
    generated: list[int] = []
    accepted_total = 0
    proposed_total = 0
    target_forwards = 0
    verify_time_s = 0.0

    start_time = time.perf_counter()
    stop_generation = False
    while len(generated) < max_new_tokens and not stop_generation:
        target_forwards += 1
        current = prefix + generated
        remaining = max_new_tokens - len(generated)
        proposed = draft_proposer(prefix, generated, min(num_draft_tokens, remaining))
        full_input = current + proposed
        verify_start = time.perf_counter()
        logits = run_target_logits(llm, torch.tensor([full_input], dtype=torch.long))
        verify_time_s += time.perf_counter() - verify_start

        accept_start = len(current)
        accepted: list[int] = []
        rejected = False
        for i, d in enumerate(proposed):
            pred = int(torch.argmax(logits[0, accept_start + i - 1]).item())
            if pred == d:
                accepted.append(d)
            else:
                accepted.append(pred)
                rejected = True
                break

        if proposed and not rejected and len(proposed) < remaining:
            # All drafts accepted and budget remains: take one bonus target token.
            bonus_pos = accept_start + len(proposed) - 1
            accepted.append(int(torch.argmax(logits[0, bonus_pos]).item()))
        elif not accepted:
            # No drafts proposed: take exactly one target token.
            accepted.append(int(torch.argmax(logits[0, accept_start - 1]).item()))

        # Append tokens one at a time so we can stop as soon as an EOS token
        # is emitted, matching the greedy baseline behavior.
        for i, tok in enumerate(accepted):
            generated.append(tok)
            if i < len(proposed):
                proposed_total += 1
                if tok == proposed[i]:
                    accepted_total += 1
            if tok in eos_ids:
                stop_generation = True
                break

        # Always enforce the generation budget, even if an EOS was produced in
        # the same step that crossed max_new_tokens.
        if len(generated) > max_new_tokens:
            generated = generated[:max_new_tokens]

        if len(generated) >= max_new_tokens or stop_generation:
            break
    elapsed = time.perf_counter() - start_time

    acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0.0
    speculative_tps = len(generated) / elapsed if elapsed > 0 else 0.0

    return {
        "token_ids": generated,
        "baseline_token_ids": [],  # caller-filled placeholder
        "bit_exact": False,  # caller-filled placeholder
        "accepted_total": accepted_total,
        "proposed_total": proposed_total,
        "acceptance_rate": acceptance_rate,
        "target_forwards": target_forwards,
        "verify_time_s": verify_time_s,
        "baseline_tps": 0.0,  # filled by CLI caller
        "speculative_tps": speculative_tps,
        "projected_tps": 0.0,  # filled by CLI caller
    }


def _load_prompts(prompt_file: str) -> list[dict[str, Any]]:
    """Load the prompt fixture as a list of prompt entries."""
    data = json.loads(Path(prompt_file).read_text())
    return list(data["prompts"])


def _expand_prompts_for_gate(
    tokenizer: Any, prompts: list[dict[str, Any]]
) -> list[tuple[str, list[int]]]:
    """Expand fixture prompts to exact context lengths for the tokenizer gate."""
    expanded: list[tuple[str, list[int]]] = []
    for item in prompts:
        seed = item["text"]
        target_len = item["context_len"]
        tokens = tokenizer.encode(seed)
        if len(tokens) == 0:
            if target_len == 0:
                expanded.append((seed, []))
                continue
            raise ValueError(
                f"Prompt {item.get('id', seed)!r} encoded to an empty token list; "
                f"cannot expand to {target_len} tokens"
            )
        if len(tokens) >= target_len:
            tokens = tokens[:target_len]
        else:
            repeated = (seed + " ") * ((target_len // len(tokens)) + 2)
            tokens = tokenizer.encode(repeated)[:target_len]
        actual_len = len(tokens)
        if actual_len != target_len:
            raise ValueError(
                f"Prompt {item.get('id', seed)!r}: expected {target_len} tokens, "
                f"got {actual_len}"
            )
        expanded.append((tokenizer.decode(tokens), tokens))
    return expanded


def _default_model_path() -> str:
    return "models/gemma-4-26B-A4B-it-AWQ-4bit"


def _default_draft_model_path() -> str:
    return "models/gemma-4-E2B-it-AWQ-INT4"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline speculative decoding prototype for Gemma4.",
    )
    parser.add_argument("--target-model", default=_default_model_path())
    parser.add_argument(
        "--draft-model",
        default=_default_draft_model_path(),
        help="Optional draft model path. If unset, n-gram drafting is used.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FIXTURE),
        help="Path to the JSON prompt fixture.",
    )
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--fail-on-mismatch",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=True,
        help="Exit with code 1 when any prompt is not bit_exact (default: true).",
    )
    return parser


def _build_memory_report() -> dict[str, Any]:
    """Measure GPU memory and return a structured memory gate report."""
    if not torch.cuda.is_available():
        return {
            "peak_reserved_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
            "free_ratio": 1.0,
            "memory_gate_passed": True,
        }

    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory_ok = (free_bytes / total_bytes) >= 0.05
    return {
        "peak_reserved_gb": peak_reserved_bytes / 1e9,
        "free_gb": free_bytes / 1e9,
        "total_gb": total_bytes / 1e9,
        "free_ratio": free_bytes / total_bytes,
        "memory_gate_passed": memory_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.isdir(args.target_model):
        print(
            f"[ERROR] target model directory not found: {args.target_model}",
            file=sys.stderr,
        )
        return 2

    raw_prompts = _load_prompts(args.prompt_file)

    target_tok = _load_tokenizer(args.target_model)
    draft_tok = _load_tokenizer(args.draft_model)
    gate_prompts = _expand_prompts_for_gate(target_tok, raw_prompts)
    gate = build_report(
        args.target_model, args.draft_model, target_tok, draft_tok, gate_prompts
    )
    if not gate["passed"]:
        print(json.dumps({"tokenizer_gate": gate}, indent=2), file=sys.stderr)
        return 2

    max_context = max(p["context_len"] for p in raw_prompts)
    max_new_tokens = max(p["max_new_tokens"] for p in raw_prompts)
    max_k = 8
    needed_model_len = max_context + max_new_tokens + max_k + 32

    torch.cuda.reset_peak_memory_stats()
    target_llm = LLM(
        model=args.target_model,
        max_model_len=needed_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
    )

    draft_llm: LLM | None = None
    try:
        torch.cuda.reset_peak_memory_stats()
        draft_llm = LLM(
            model=args.draft_model,
            max_model_len=needed_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
        )
    except torch.cuda.OutOfMemoryError as e:
        memory_report = {
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "free_gb": 0.0,
            "total_gb": 0.0,
            "free_ratio": 0.0,
            "memory_gate_passed": False,
            "error": str(e),
        }
        print(json.dumps({"memory_gate": memory_report}, indent=2), file=sys.stderr)
        target_llm.shutdown()
        return 2

    memory_report = _build_memory_report()
    if not memory_report["memory_gate_passed"]:
        print(json.dumps({"memory_gate": memory_report}, indent=2), file=sys.stderr)
        target_llm.shutdown()
        if draft_llm is not None:
            draft_llm.shutdown()
        return 2

    try:
        if args.draft_model and os.path.isdir(args.draft_model):

            def draft_proposer(
                prefix: list[int], generated: list[int], k: int
            ) -> list[int]:
                context_ids = prefix + generated
                return generate_draft_tokens_from_ids(draft_llm, context_ids, k)
        else:

            def draft_proposer(
                prefix: list[int], generated: list[int], k: int
            ) -> list[int]:
                return propose_ngram(
                    prefix, generated, k, args.ngram_min, args.ngram_max
                )

        prompt_results: list[dict[str, Any]] = []
        all_bit_exact = True
        total_accepted = 0
        total_proposed = 0
        total_decode_tokens = 0
        total_decode_time = 0.0

        for prompt_item in raw_prompts:
            prompt_text = prompt_item["text"]
            per_prompt_max_new_tokens = prompt_item["max_new_tokens"]

            baseline_result = baseline_greedy(
                target_llm, prompt_text, per_prompt_max_new_tokens
            )
            prompt_token_ids = baseline_result["prompt_token_ids"]
            baseline_token_ids = baseline_result["token_ids"]

            spec_result = speculative_decode(
                target_llm,
                draft_proposer,
                prompt_token_ids,
                max_new_tokens=per_prompt_max_new_tokens,
                num_draft_tokens=args.num_draft_tokens,
            )

            spec_result["baseline_token_ids"] = baseline_token_ids
            bit_exact = spec_result["token_ids"] == baseline_token_ids
            spec_result["bit_exact"] = bit_exact
            if not bit_exact:
                all_bit_exact = False

            total_accepted += spec_result["accepted_total"]
            total_proposed += spec_result["proposed_total"]

            baseline_tps = baseline_result["decode_tps"]
            spec_result["baseline_tps"] = baseline_tps

            target_forwards = spec_result["target_forwards"]
            if target_forwards > 0:
                theoretical_speedup = (
                    1.0 + spec_result["accepted_total"] / target_forwards
                )
            else:
                theoretical_speedup = 1.0
            spec_result["projected_tps"] = baseline_tps * theoretical_speedup

            if args.draft_model and os.path.isdir(args.draft_model):
                # Effective TPS is accumulated globally, not per prompt.
                total_decode_tokens += spec_result["accepted_total"] + target_forwards
                total_decode_time += spec_result["verify_time_s"]

            prompt_results.append(
                {
                    "id": prompt_item["id"],
                    "prompt_text": prompt_text,
                    "context_len": prompt_item["context_len"],
                    "max_new_tokens": per_prompt_max_new_tokens,
                    "baseline_tokens": baseline_token_ids,
                    "speculative_tokens": spec_result["token_ids"],
                    "bit_exact": bit_exact,
                    "accepted_total": spec_result["accepted_total"],
                    "proposed_total": spec_result["proposed_total"],
                    "acceptance_rate": spec_result["acceptance_rate"],
                    "target_forwards": spec_result["target_forwards"],
                    "baseline_prefill_time_s": baseline_result["prefill_time_s"],
                    "baseline_decode_time_s": baseline_result["decode_time_s"],
                    "baseline_decode_tps": baseline_result["decode_tps"],
                    "speculative_tps": spec_result["speculative_tps"],
                    "projected_tps": spec_result["projected_tps"],
                }
            )

        aggregate_acceptance_rate = (
            total_accepted / total_proposed if total_proposed > 0 else 0.0
        )
        mean_baseline_decode_tps = (
            sum(p["baseline_decode_tps"] for p in prompt_results) / len(prompt_results)
            if prompt_results
            else 0.0
        )
        effective_tps = (
            total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0
        )

        output = {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "num_draft_tokens": args.num_draft_tokens,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "memory_report": memory_report,
            "tokenizer_gate": gate,
            "summary": {
                "total_prompts": len(prompt_results),
                "bit_exact_all": all_bit_exact,
                "mean_acceptance_rate": aggregate_acceptance_rate,
                "mean_baseline_decode_tps": mean_baseline_decode_tps,
                "effective_tps": effective_tps,
            },
            "prompts": prompt_results,
        }

        print(json.dumps(output, indent=2, sort_keys=True))

        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n"
            )

        if args.fail_on_mismatch and not all_bit_exact:
            print("[ERROR] speculative output does not match baseline", file=sys.stderr)
            return 1
        return 0
    finally:
        target_llm.shutdown()
        if draft_llm is not None:
            draft_llm.shutdown()


if __name__ == "__main__":
    sys.exit(main())
