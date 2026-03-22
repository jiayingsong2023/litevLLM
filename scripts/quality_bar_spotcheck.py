#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tier-B (docs/INFERENCE_ACCURACY.md §1/§6) spot-check: completions via LiteEngine with SamplingParams,
with heuristics aligned to readability / coherence / first-token sanity. No HF reference.

LiteEngine applies temperature / top_p / top_k / repetition_penalty when temperature > 0;
use --temperature 0 for strictly greedy decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# Repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FASTINFERENCE_KV_FP8", "0")

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig  # noqa: E402
from vllm.engine.lite_engine import LiteEngine  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

# INFERENCE_ACCURACY.md §6: temperature 0～0.7 for natural-ish text
_MAX_DOC_TEMPERATURE = 0.7

# Heuristic thresholds (tune for obvious garbage only)
_REPLACEMENT_CHARS_SEVERE = 0.12
_CONTROL_CHAR_RATIO_SEVERE = 0.08
_MAX_TOKEN_REPEAT_RATIO_SEVERE = 0.82
_MAX_CHAR_RUN_SEVERE = 24


def _lite_engine_step_budget(engine: LiteEngine, prompt_token_len: int, max_new_tokens: int) -> int:
    chunk_sz = max(1, int(getattr(engine, "_prefill_chunk_size", 512)))
    prefill_chunks = max(1, (prompt_token_len + chunk_sz - 1) // chunk_sz)
    return prefill_chunks + max_new_tokens * 3 + 500


def _run_lite_steps_until(
    engine: LiteEngine,
    description: str,
    max_steps: int,
    stop_fn: Callable[[List[Any]], Optional[Any]],
) -> Any:
    for _ in range(max_steps):
        step_outputs = engine.step()
        done = stop_fn(step_outputs)
        if done is not None:
            return done
    raise RuntimeError(
        f"{description}: exceeded {max_steps} LiteEngine.step() calls "
        f"(active_request_count={engine.active_request_count})."
    )


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    cfg_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    try:
        if not os.path.isfile(cfg_path):
            return group_size, bits
        with open(cfg_path, "r") as f:
            raw = json.load(f)
        qc = raw.get("quantization_config") or {}
        groups = qc.get("config_groups")
        if isinstance(groups, dict):
            for g in groups.values():
                if not isinstance(g, dict):
                    continue
                w = g.get("weights")
                if isinstance(w, dict):
                    if w.get("group_size") is not None:
                        group_size = int(w["group_size"])
                    if w.get("num_bits") is not None:
                        bits = int(w["num_bits"])
                    break
        if qc.get("group_size") is not None:
            group_size = int(qc["group_size"])
        if qc.get("bits") is not None:
            bits = int(qc["bits"])
    except Exception as e:
        sys.stderr.write(f"[Warning] AWQ config parse failed: {e}\n")
    return group_size, bits


# Mirrors docs/INFERENCE_ACCURACY.md §6 prompt list
DEFAULT_SPOTCHECK_PROMPTS: List[Tuple[str, str]] = [
    ("en_capital", "The capital of France is"),
    ("en_bst", "Explain what a binary search tree is in one short paragraph."),
    ("en_python_sum", "Write a Python function that returns the sum of a list of integers."),
    ("zh_capital", "法国的首都是"),
    ("zh_gd", "用两三句话解释什么是梯度下降。"),
    ("zh_hello", "请用 Python 写一个简单的 Hello World 程序。"),
    (
        "code_fib",
        'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    ',
    ),
    ("json_partial", '{"name": "test", '),
    ("short_hi", "Hi,"),
]

# Minimal set: doc asks for at least 3–5 prompts; covers EN / ZH / short boundary
MINIMAL_PROMPT_IDS = frozenset(
    {"en_capital", "en_bst", "zh_capital", "zh_gd", "short_hi"}
)


def _load_prompts_from_file(path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    out.append((f"file_{i}", item))
                elif isinstance(item, dict) and "prompt" in item:
                    pid = str(item.get("id", f"file_{i}"))
                    out.append((pid, str(item["prompt"])))
        else:
            raise ValueError("JSON prompts file must be a list of strings or objects with 'prompt'.")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append((f"line_{i}", line))
    return out


def _max_consecutive_same_token(token_ids: List[int]) -> int:
    if not token_ids:
        return 0
    best = 1
    run = 1
    prev = token_ids[0]
    for t in token_ids[1:]:
        if t == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
            prev = t
    return best


def _max_consecutive_same_char(text: str) -> int:
    if not text:
        return 0
    best = 1
    run = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
            prev = ch
    return best


def _control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        o = ord(ch)
        if o < 32 and ch not in "\n\r\t":
            bad += 1
    return bad / len(text)


def _first_token_sanity(
    tokenizer: Any,
    first_token_id: Optional[int],
) -> Tuple[bool, List[str]]:
    """Return (ok, messages). ok=False => first-token heuristic suspects garbage."""
    msgs: List[str] = []
    if first_token_id is None:
        msgs.append("no_first_token")
        return False, msgs
    try:
        piece = tokenizer.decode([first_token_id], skip_special_tokens=False)
    except Exception as e:
        msgs.append(f"decode_first_token_failed: {e}")
        return False, msgs
    if piece.count("\ufffd") == len(piece) and len(piece) > 0:
        msgs.append("first_token_decodes_only_replacement")
        return False, msgs
    # Heuristic: many unusual private-use / control chars in a single-token decode
    if len(piece) <= 6 and _control_char_ratio(piece) > 0.5:
        msgs.append("first_token_high_control_ratio")
        return False, msgs
    return True, msgs


def analyze_tier_b(
    text: str,
    token_ids: List[int],
    tokenizer: Any,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Map outputs to INFERENCE_ACCURACY.md tier-B dimensions:
    1 readability, 2 coherence, 4 first-token (3 relevance is not automated).

    Returns (severe_any, detail_dict, flat_messages).
    """
    detail: Dict[str, Any] = {
        "readability": {"pass": True, "notes": []},
        "coherence": {"pass": True, "notes": []},
        "first_token": {"pass": True, "notes": []},
    }
    flat: List[str] = []
    severe = False

    n = len(text)
    repl = text.count("\ufffd")
    repl_ratio = repl / max(n, 1)
    if repl_ratio >= _REPLACEMENT_CHARS_SEVERE:
        detail["readability"]["pass"] = False
        detail["readability"]["notes"].append(
            f"replacement_char_ratio={repl_ratio:.3f}>={_REPLACEMENT_CHARS_SEVERE}"
        )
        flat.append(detail["readability"]["notes"][-1])
        severe = True

    cc = _control_char_ratio(text)
    if cc >= _CONTROL_CHAR_RATIO_SEVERE:
        detail["readability"]["pass"] = False
        detail["readability"]["notes"].append(f"control_char_ratio={cc:.3f}>={_CONTROL_CHAR_RATIO_SEVERE}")
        flat.append(detail["readability"]["notes"][-1])
        severe = True

    if n == 0 and len(token_ids) == 0:
        detail["readability"]["pass"] = False
        detail["readability"]["notes"].append("empty_completion")
        flat.append("empty_completion")
        severe = True

    mct = _max_consecutive_same_token(token_ids)
    if token_ids:
        rep_r = mct / len(token_ids)
        if rep_r >= _MAX_TOKEN_REPEAT_RATIO_SEVERE:
            detail["coherence"]["pass"] = False
            detail["coherence"]["notes"].append(
                f"token_repeat_run={mct} ratio={rep_r:.2f}>={_MAX_TOKEN_REPEAT_RATIO_SEVERE}"
            )
            flat.append(detail["coherence"]["notes"][-1])
            severe = True

    mcr = _max_consecutive_same_char(text)
    if mcr >= _MAX_CHAR_RUN_SEVERE:
        detail["coherence"]["pass"] = False
        detail["coherence"]["notes"].append(
            f"char_repeat_run={mcr}>={_MAX_CHAR_RUN_SEVERE}"
        )
        flat.append(detail["coherence"]["notes"][-1])
        severe = True

    first_id = token_ids[0] if token_ids else None
    ft_ok, ft_msgs = _first_token_sanity(tokenizer, first_id)
    if not ft_ok:
        detail["first_token"]["pass"] = False
        detail["first_token"]["notes"].extend(ft_msgs)
        flat.extend(ft_msgs)
        severe = True

    detail["tier_b_alignment"] = {
        "readability_ok": detail["readability"]["pass"],
        "coherence_ok": detail["coherence"]["pass"],
        "first_token_ok": detail["first_token"]["pass"],
        # relevance_ok: not computed (needs reference or embeddings)
    }
    return severe, detail, flat


def _build_engine(model_path: str, quant: str) -> LiteEngine:
    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    q_cfg = None
    if quant == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        gs, wb = _read_awq_group_size_and_bits(model_path)
        q_cfg = AWQConfig(weight_bits=wb, group_size=gs)
        sys.stderr.write(f"[Setup] AWQ group_size={gs}, weight_bits={wb}\n")
    elif quant == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        q_cfg = GGUFConfig()
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    return LiteEngine(v_cfg)


def _resolve_prompts(
    args: argparse.Namespace,
) -> List[Tuple[str, str]]:
    if args.prompts_file:
        return _load_prompts_from_file(args.prompts_file)
    prompts = list(DEFAULT_SPOTCHECK_PROMPTS)
    if args.prompt_subset == "minimal":
        prompts = [p for p in prompts if p[0] in MINIMAL_PROMPT_IDS]
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier-B spot-check per docs/INFERENCE_ACCURACY.md (Lite only, no HF). "
        "See docs/INFERENCE_ACCURACY.md for goals."
    )
    parser.add_argument("--model", type=str, required=True, help="Model directory (same as LiteEngine).")
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gguf"])
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens per prompt.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="0 = greedy argmax; (0, 0.7] = multinomial after scaling (INFERENCE_ACCURACY.md §6). Default 0.7 for natural-ish text.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling when temperature > 0 (ignored for greedy). Default 0.95.",
    )
    parser.add_argument(
        "--prompt-subset",
        type=str,
        choices=("full", "minimal"),
        default="full",
        help="minimal: 5 prompts (doc: at least 3–5); full: all 9 built-in prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional .json list or text file with one prompt per line (# comments ok). Overrides --prompt-subset.",
    )
    parser.add_argument(
        "--no-heuristics-fail",
        action="store_true",
        help="Do not exit non-zero when heuristics flag severe issues (still print WARN).",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object per line to stdout.")
    args = parser.parse_args()

    if args.temperature < 0 or args.temperature > _MAX_DOC_TEMPERATURE:
        sys.stderr.write(
            f"[Warning] INFERENCE_ACCURACY.md §6 recommends temperature in [0, {_MAX_DOC_TEMPERATURE}]; "
            f"got {args.temperature}. Clamping to range for reporting.\n"
        )
        args.temperature = max(0.0, min(float(args.temperature), _MAX_DOC_TEMPERATURE))

    if args.temperature <= 1e-6:
        sys.stderr.write(
            "[Note] Greedy decoding (--temperature 0): top_p has no effect on token choice.\n"
        )

    prompts = _resolve_prompts(args)
    if not prompts:
        sys.stderr.write("[Error] No prompts to run (empty subset or file).\n")
        return 2

    sys.stderr.write(
        f"[Tier-B] model={args.model!r} quant={args.quant} "
        f"prompts={len(prompts)} subset={args.prompt_subset} max_new_tokens={args.max_new_tokens}\n"
    )
    engine = _build_engine(args.model, args.quant)
    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    engine.tokenizer = tokenizer

    any_severe = False
    summary_rows: List[dict] = []

    for pid, prompt in prompts:
        sp = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        req_id = f"qb_{pid}"
        engine.add_request(req_id, prompt, sp)
        enc = tokenizer.encode(prompt)
        prompt_len = len(enc)
        budget = _lite_engine_step_budget(engine, prompt_len, args.max_new_tokens)

        def _done(outs: List[Any]) -> Optional[Any]:
            if not outs:
                return None
            ro = outs[0]
            if ro.finished and ro.outputs:
                return ro.outputs[0]
            return None

        try:
            out = _run_lite_steps_until(engine, f"spotcheck[{pid}]", budget, _done)
        except RuntimeError as e:
            sys.stderr.write(f"[ERROR] {pid}: {e}\n")
            any_severe = True
            summary_rows.append({"id": pid, "error": str(e)})
            continue

        text = out.text if out else ""
        token_ids = list(out.token_ids) if out else []
        first_tok = token_ids[0] if token_ids else None
        severe, detail, msgs = analyze_tier_b(text, token_ids, tokenizer)

        row = {
            "id": pid,
            "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
            "first_token_id": first_tok,
            "num_tokens": len(token_ids),
            "text": text,
            "heuristic_warn": msgs,
            "heuristic_severe": severe,
            "tier_b_detail": detail,
        }
        summary_rows.append(row)

        if args.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            print("\n" + "=" * 72)
            print(f"[{pid}] first_token_id={first_tok} num_tokens={len(token_ids)}")
            ta = detail.get("tier_b_alignment", {})
            print(
                f"  tier_b: readability={ta.get('readability_ok')} "
                f"coherence={ta.get('coherence_ok')} first_token={ta.get('first_token_ok')}"
            )
            if msgs:
                print(f"  heuristics ({'WARN' if severe else 'info'}): {', '.join(msgs)}")
            print(f"  output:\n{text!r}")

        if severe:
            any_severe = True

    if not args.json:
        passed = sum(
            1
            for r in summary_rows
            if "tier_b_detail" in r
            and r["tier_b_detail"].get("tier_b_alignment", {}).get("readability_ok")
            and r["tier_b_detail"].get("tier_b_alignment", {}).get("coherence_ok")
            and r["tier_b_detail"].get("tier_b_alignment", {}).get("first_token_ok")
        )
        total = len([r for r in summary_rows if "error" not in r])
        sys.stderr.write("\n" + "=" * 72 + "\n")
        sys.stderr.write(
            f"Summary: {passed}/{total} cases passed automated tier-B heuristics "
            f"(readability+coherence+first_token). Human review still required.\n"
        )

    if any_severe and not args.no_heuristics_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
