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

from vllm import LLM
from vllm.sampling_params import SamplingParams


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
    """Return the set of EOS token IDs used by the model.

    Tries the tokenizer first, then falls back to the HF model config (e.g.
    Gemma4 exposes extra turn/EOS ids in ``generation_config.json`` that the
    tokenizer does not surface).  If neither source is available, returns an
    empty set so generation is only bounded by ``max_new_tokens``.
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

    # The engine's greedy baseline also honors eos_token_id from the HF config.
    engine = getattr(llm, "engine", None)
    model_config = (
        getattr(engine, "model_config", None) if engine is not None else None
    )
    hf_config = (
        getattr(model_config, "hf_config", None)
        if model_config is not None
        else None
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


def baseline_greedy(
    llm: LLM, prompt_text: str, max_new_tokens: int
) -> tuple[list[int], list[int]]:
    """Greedy baseline run. Returns (prompt_token_ids, generated_token_ids).

    Uses the engine's step loop directly (rather than ``LLM.generate``) so that
    Gemma4's prefill-only first step, which produces no outputs, does not abort
    the run.
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
    while llm.engine.active_request_count > 0 and step_count < step_budget:
        step_count += 1
        outs = llm.engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            if out.finished:
                final_output = out

    if final_output is None:
        raise RuntimeError(
            f"baseline greedy did not finish within step_budget={step_budget}; "
            f"active_request_count={llm.engine.active_request_count}"
        )

    return list(final_output.prompt_token_ids), list(
        final_output.outputs[0].token_ids
    )


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

    start_time = time.perf_counter()
    stop_generation = False
    while len(generated) < max_new_tokens and not stop_generation:
        target_forwards += 1
        current = prefix + generated
        proposed = draft_proposer(prefix, generated, num_draft_tokens)
        full_input = current + proposed
        logits = run_target_logits(llm, torch.tensor([full_input], dtype=torch.long))

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

        if proposed and not rejected:
            # All drafts accepted: take one bonus target token.
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

        if len(generated) >= max_new_tokens and not stop_generation:
            generated = generated[:max_new_tokens]
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
        "baseline_tps": 0.0,  # filled by CLI caller
        "speculative_tps": speculative_tps,
        "projected_tps": 0.0,  # filled by CLI caller
    }


def _default_model_path() -> str:
    return "models/gemma-4-26B-A4B-it-AWQ-4bit"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline n-gram speculative decoding prototype for Gemma4.",
    )
    parser.add_argument("--target-model", default=_default_model_path())
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        prompt_text = args.prompt

    target_model_path = args.target_model
    if not os.path.isdir(target_model_path):
        print(
            f"[ERROR] target model directory not found: {target_model_path}",
            file=sys.stderr,
        )
        return 2

    target_llm = LLM(
        model=target_model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=1024,
    )

    baseline_start = time.perf_counter()
    prompt_token_ids, baseline_token_ids = baseline_greedy(
        target_llm, prompt_text, args.max_new_tokens
    )
    baseline_elapsed = time.perf_counter() - baseline_start

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        return propose_ngram(prefix, generated, k, args.ngram_min, args.ngram_max)

    spec_result = speculative_decode(
        target_llm,
        draft_proposer,
        prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
    )

    baseline_tps = (
        len(baseline_token_ids) / baseline_elapsed if baseline_elapsed > 0 else 0.0
    )
    spec_result["baseline_token_ids"] = baseline_token_ids
    spec_result["bit_exact"] = spec_result["token_ids"] == baseline_token_ids
    spec_result["baseline_tps"] = baseline_tps

    target_forwards = spec_result["target_forwards"]
    if target_forwards > 0:
        theoretical_speedup = 1.0 + spec_result["accepted_total"] / target_forwards
    else:
        theoretical_speedup = 1.0
    spec_result["projected_tps"] = baseline_tps * theoretical_speedup

    output = {
        "prompt_text": prompt_text,
        "target_model": target_model_path,
        "num_draft_tokens": args.num_draft_tokens,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "baseline_tokens": baseline_token_ids,
        "speculative_tokens": spec_result["token_ids"],
        "bit_exact": spec_result["bit_exact"],
        "accepted_total": spec_result["accepted_total"],
        "proposed_total": spec_result["proposed_total"],
        "acceptance_rate": spec_result["acceptance_rate"],
        "baseline_tps": spec_result["baseline_tps"],
        "speculative_tps": spec_result["speculative_tps"],
        "projected_tps": spec_result["projected_tps"],
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n"
        )

    target_llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
