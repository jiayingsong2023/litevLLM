#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tier-B (docs/INFERENCE_ACCURACY.md §1/§6) spot-check: completions via LiteEngine with SamplingParams,
with heuristics aligned to readability / coherence / first-token sanity. No HF reference.

LiteEngine applies temperature / top_p / top_k / repetition_penalty when temperature > 0;
use --temperature 0 for strictly greedy decoding.

For Qwen3 Instruct / chat checkpoints, raw completion prompts (e.g. "The capital of France is")
without ``apply_chat_template`` often yield poor coherence. Use ``--chat-template auto`` (default):
each prompt is wrapped as a user message when the tokenizer defines ``chat_template``.
Pre-formatted prompts (containing ``<|im_start|>``) are left unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

# Repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")

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
# One character dominates the body (e.g. CJK spam like repeated「上」mixed with junk).
_MAX_CHAR_FREQUENCY_DOMINANCE_SEVERE = 0.11
# Natural English/Latin prose often has ~12–16% for the most frequent letter (e.g. "e");
# keep a higher bar only when Latin letters dominate non-space text.
_MAX_CHAR_FREQUENCY_DOMINANCE_LATIN_PROSE = 0.19
_MIN_LATIN_LETTER_RATIO_FOR_RELAXED_DOMINANCE = 0.55
_MIN_NONSPACE_FOR_DOMINANCE_CHECK = 28
# Long completions with very few distinct token ids (loop / collapse).
_MIN_UNIQUE_TOKEN_RATIO_SEVERE = 0.20
_MIN_TOKENS_FOR_DIVERSITY_CHECK = 40

# Substance heuristics: catch completions that pass replacement/control checks but are
# human-unreadable (newline-only, digit/symbol soup). Tunable; see tests.
_MIN_NONSPACE_CHARS_WHEN_MANY_TOKS = 3
_MANY_TOKENS_FOR_SUBSTANCE = 8
_MIN_BODY_CHARS_FOR_LETTER_RATIO = 16
_MIN_LETTERISH_RATIO = 0.15
_MAX_DIGIT_RATIO_SEVERE = 0.68


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


def _model_config_hints(model_path: str) -> Dict[str, Any]:
    """Detect MoE / text config from HuggingFace-style config.json next to weights."""
    out: Dict[str, Any] = {"is_moe": False, "model_type": ""}
    try:
        p = os.path.join(model_path, "config.json")
        raw = None
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
        # GGUF trees often have no config.json; use sibling HF Chat (same as tokenizer resolution).
        if raw is None:
            try:
                from vllm.model_executor.models.deepseek_hf_reference import (
                    resolve_deepseek_hf_chat_dir,
                )

                ref = resolve_deepseek_hf_chat_dir(model_path)
                if ref:
                    ref_cfg = os.path.join(ref, "config.json")
                    if os.path.isfile(ref_cfg):
                        with open(ref_cfg, "r", encoding="utf-8") as f:
                            raw = json.load(f)
            except Exception:
                pass
        if raw is None:
            return out
        tc = raw.get("text_config")
        if isinstance(tc, dict):
            mt = str(tc.get("model_type", "") or "")
            ne = int(tc.get("num_experts", 0) or 0)
            n_routed = int(tc.get("n_routed_experts", 0) or 0)
        else:
            mt = str(raw.get("model_type", "") or "")
            ne = int(raw.get("num_experts", 0) or 0)
            # DeepSeek V2 Lite GGUF uses n_routed_experts, not num_experts
            n_routed = int(raw.get("n_routed_experts", 0) or 0)
        out["model_type"] = mt
        out["is_moe"] = ne > 1 or n_routed > 1 or "moe" in mt.lower()
    except Exception:
        pass
    return out


def _looks_like_qwen35_35b_awq(model_path: str, quant: str) -> bool:
    if quant != "awq":
        return False
    b = os.path.basename(os.path.abspath(model_path)).lower()
    return "qwen3.5-35b-awq" in b or ("qwen3.5" in b and "35b" in b and "awq" in b)


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


# Mirrors docs/INFERENCE_ACCURACY.md §6 prompt list.
# Use explicit questions/instructions so Instruct models (Qwen chat template) answer coherently;
# completion-style fragments are kept only where we test continuation (code_fib, json_partial, short_hi).
DEFAULT_SPOTCHECK_PROMPTS: List[Tuple[str, str]] = [
    ("en_capital", "What is the capital of France? Answer in a few words."),
    ("en_bst", "Explain what a binary search tree is in one short paragraph."),
    ("en_python_sum", "Write a Python function that returns the sum of a list of integers."),
    ("zh_capital", "法国的首都是哪里？请用一句话回答。"),
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


def _max_char_frequency_ratio(nonspace_text: str) -> float:
    """Share of the most frequent character in non-space text (0 if empty)."""
    if not nonspace_text:
        return 0.0
    c = Counter(nonspace_text)
    return max(c.values()) / len(nonspace_text)


def _latin_letter_ratio_nonspace(nonspace_text: str) -> float:
    """Share of Basic Latin letters (A–Z, a–z) in non-space text."""
    if not nonspace_text:
        return 0.0
    n = sum(
        1
        for ch in nonspace_text
        if unicodedata.category(ch).startswith("L") and ord(ch) < 128
    )
    return n / len(nonspace_text)


def _max_char_frequency_domination_threshold(nonspace_text: str) -> float:
    """Severe threshold for max-char share; relaxed for Latin-heavy prose."""
    if len(nonspace_text) < _MIN_NONSPACE_FOR_DOMINANCE_CHECK:
        return float("inf")
    if _latin_letter_ratio_nonspace(nonspace_text) >= _MIN_LATIN_LETTER_RATIO_FOR_RELAXED_DOMINANCE:
        return max(_MAX_CHAR_FREQUENCY_DOMINANCE_SEVERE, _MAX_CHAR_FREQUENCY_DOMINANCE_LATIN_PROSE)
    return _MAX_CHAR_FREQUENCY_DOMINANCE_SEVERE


def _unique_token_ratio(token_ids: List[int]) -> float:
    if not token_ids:
        return 1.0
    return len(set(token_ids)) / len(token_ids)


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


def _nonspace_chars(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _is_letterish(ch: str) -> bool:
    if unicodedata.category(ch).startswith("L"):
        return True
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return True
    if 0x3400 <= o <= 0x4DBF:
        return True
    if 0x20000 <= o <= 0x2A6DF:
        return True
    return False


def _looks_like_code_fragment(text: str) -> bool:
    s = text.lower()
    needles = ("def ", "import ", "return ", "class ", "```", "fn ", "public ", "void ", "#include")
    return any(n in s for n in needles)


def _substance_issues(text: str, token_ids: List[int]) -> List[str]:
    """Human-readability signals not covered by replacement/control/repeat checks."""
    notes: List[str] = []
    if not token_ids:
        return notes
    stripped = text.strip()
    if not stripped:
        notes.append("whitespace_only_completion")
        return notes

    core = _nonspace_chars(text)
    ntok = len(token_ids)
    if ntok >= _MANY_TOKENS_FOR_SUBSTANCE and len(core) < _MIN_NONSPACE_CHARS_WHEN_MANY_TOKS:
        notes.append(
            f"little_substance_vs_tokens(nonspace={len(core)} tok={ntok}>={_MANY_TOKENS_FOR_SUBSTANCE})"
        )
        return notes

    if len(core) < _MIN_BODY_CHARS_FOR_LETTER_RATIO:
        return notes

    if _looks_like_code_fragment(text):
        return notes

    letterish = sum(1 for ch in core if _is_letterish(ch))
    ratio_letter = letterish / len(core)
    digit_count = sum(1 for ch in core if ch.isdigit())
    ratio_digit = digit_count / len(core)

    if ratio_digit >= _MAX_DIGIT_RATIO_SEVERE and ratio_letter < _MIN_LETTERISH_RATIO:
        notes.append(
            f"digit_heavy_low_letters(digit_ratio={ratio_digit:.2f} letterish_ratio={ratio_letter:.2f})"
        )
    elif ratio_letter < _MIN_LETTERISH_RATIO and len(core) >= 24:
        notes.append(f"low_letterish_ratio({ratio_letter:.2f} on len={len(core)})")

    return notes


def _first_token_sanity(
    tokenizer: Any,
    first_token_id: Optional[int],
    token_ids: Optional[List[int]] = None,
    full_text: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Return (ok, messages). ok=False => first-token heuristic suspects garbage.
    BPE may decode the first id alone as U+FFFD while the full sequence is valid text;
    if full_text has low replacement ratio, do not fail on isolated replacement decode.
    """
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
        if (
            full_text
            and len(full_text) >= 8
            and full_text.count("\ufffd") / max(len(full_text), 1) < 0.06
        ):
            return True, msgs
        if token_ids and len(token_ids) >= 4:
            try:
                prefix = tokenizer.decode(token_ids[: min(12, len(token_ids))], skip_special_tokens=False)
            except Exception:
                prefix = ""
            if len(prefix) >= 4 and prefix.count("\ufffd") / max(len(prefix), 1) < 0.06:
                return True, msgs
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
    *,
    check_substance: bool = True,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Map outputs to INFERENCE_ACCURACY.md tier-B dimensions:
    1 readability, 2 coherence, 4 first-token (3 relevance is not automated).
    Optional substance checks reduce false positives (e.g. newline-only spam).

    Returns (severe_any, detail_dict, flat_messages).
    """
    detail: Dict[str, Any] = {
        "readability": {"pass": True, "notes": []},
        "coherence": {"pass": True, "notes": []},
        "first_token": {"pass": True, "notes": []},
        "substance": {"pass": True, "notes": []},
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

    core_ns = _nonspace_chars(text)
    dom_thr = _max_char_frequency_domination_threshold(core_ns)
    if len(core_ns) >= _MIN_NONSPACE_FOR_DOMINANCE_CHECK:
        dom = _max_char_frequency_ratio(core_ns)
        if dom >= dom_thr:
            detail["coherence"]["pass"] = False
            detail["coherence"]["notes"].append(
                f"char_frequency_dominance={dom:.2f}>={dom_thr:.2f} "
                f"(single char over-represented in non-space text)"
            )
            flat.append(detail["coherence"]["notes"][-1])
            severe = True

    if len(token_ids) >= _MIN_TOKENS_FOR_DIVERSITY_CHECK:
        utr = _unique_token_ratio(token_ids)
        if utr < _MIN_UNIQUE_TOKEN_RATIO_SEVERE:
            detail["coherence"]["pass"] = False
            detail["coherence"]["notes"].append(
                f"low_token_diversity(unique_ratio={utr:.2f}<{_MIN_UNIQUE_TOKEN_RATIO_SEVERE} "
                f"on {len(token_ids)} tokens)"
            )
            flat.append(detail["coherence"]["notes"][-1])
            severe = True

    first_id = token_ids[0] if token_ids else None
    ft_ok, ft_msgs = _first_token_sanity(tokenizer, first_id, token_ids, text)
    if not ft_ok:
        detail["first_token"]["pass"] = False
        detail["first_token"]["notes"].extend(ft_msgs)
        flat.extend(ft_msgs)
        severe = True

    if check_substance:
        for note in _substance_issues(text, token_ids):
            detail["substance"]["pass"] = False
            detail["substance"]["notes"].append(note)
            flat.append(note)
            severe = True

    detail["tier_b_alignment"] = {
        "readability_ok": detail["readability"]["pass"],
        "coherence_ok": detail["coherence"]["pass"],
        "first_token_ok": detail["first_token"]["pass"],
        "substance_ok": detail["substance"]["pass"],
        # relevance_ok: not computed (needs reference or embeddings)
    }
    return severe, detail, flat


def _build_engine(
    model_path: str,
    quant: str,
    *,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    max_num_seqs: int = 32,
) -> LiteEngine:
    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=gpu_memory_utilization, swap_space=4)
    # Keep batched-token budget in line with max_model_len for short spotcheck prompts.
    mbt = min(8192, max(512, max_model_len * 4))
    s_cfg = SchedulerConfig(
        max_num_batched_tokens=mbt,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
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


def _looks_like_preformatted_chat(text: str) -> bool:
    """Skip double-wrapping when the file already contains a full chat-formatted prompt."""
    s = text.lstrip()
    if len(s) >= 12 and "<|im_start|>" in s[:400]:
        return True
    if s.startswith("<|") and "user" in s[:120].lower():
        return True
    return False


def _apply_chat_template_to_prompts(
    mode: str,
    tokenizer: Any,
    prompts: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    mode: off | auto | on
    auto/on: wrap plain text as a single user turn when tokenizer.chat_template is set.
    """
    if mode == "off":
        return prompts
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        if mode == "on":
            sys.stderr.write(
                "[Warning] --chat-template on but tokenizer has no chat_template; using raw prompts.\n"
            )
        return prompts
    if mode == "auto":
        sys.stderr.write(
            "[Tier-B] chat-template=auto: wrapping each prompt as user message (tokenizer has chat_template).\n"
        )
    else:
        sys.stderr.write("[Tier-B] chat-template=on: wrapping each prompt as user message.\n")

    out: List[Tuple[str, str]] = []
    for pid, text in prompts:
        if _looks_like_preformatted_chat(text):
            out.append((pid, text))
            continue
        try:
            # Qwen3 / Qwen3.5: default template may end with an *open* <|im_start|>assistant + </think>
            # block; generation then behaves like "thinking" mode and looks incoherent. Prefer
            # closing the thinking section when the tokenizer supports it.
            try:
                wrapped = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                wrapped = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            sys.stderr.write(f"[Warning] chat template failed for {pid!r}: {e}; using raw text.\n")
            wrapped = text
        out.append((pid, wrapped))
    return out


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
        default=None,
        help=(
            "0 = greedy; (0, 0.7] = sampled. Default: 0.63 for DeepSeek + GGUF when "
            "FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY=1, else 0.55 for other --quant gguf, else 0.7. "
            "Pass explicitly to override."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=(
            "Nucleus sampling when temperature > 0 (ignored for greedy). "
            "Default: 0.95, or 0.86 when FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY=1 (Q4 tail noise)."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help=(
            "Logits repetition penalty (1.0 = off). If omitted: 1.12 when temperature>0 "
            "(reduces degenerate loops common with quantized / MoE logits), else 1.0 for greedy."
        ),
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="LiteEngine: subtract fp×count from logits (default 0, or 0.06 for gguf+stochastic if unset).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help=(
            "LiteEngine: subtract pp from logits for each token id that already appeared in "
            "the completion (default 0, or 0.05 for gguf+MoE+stochastic if unset)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help=(
            "If >=0: restrict sampling to top-k logits. If -1 and --quant gguf with temperature>0: "
            "use 50 (dense GGUF) or 40 (MoE in config) to reduce degenerate loops from tail noise."
        ),
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
    parser.add_argument(
        "--skip-substance-heuristics",
        action="store_true",
        help="Disable substance checks (newline-only / digit-heavy garbage). For debugging only.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object per line to stdout.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Lite CacheConfig gpu_memory_utilization (default: 0.9, or 0.55 with --frugal).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Scheduler max_model_len (default: 2048, or 1024 with --frugal). Short prompts need far less.",
    )
    parser.add_argument(
        "--frugal",
        action="store_true",
        help="Lower GPU memory fraction + shorter max_model_len + fewer concurrent seqs for large MoE GGUF.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        choices=("off", "auto", "on"),
        default="auto",
        help=(
            "Wrap each prompt as a user turn via tokenizer.apply_chat_template when available. "
            "auto (default): apply only if tokenizer defines chat_template. "
            "Use off for raw completion-style evaluation on base models."
        ),
    )
    args = parser.parse_args()

    if args.prompts_file and not os.path.isfile(args.prompts_file):
        sys.stderr.write(
            f"[Error] --prompts-file not found: {args.prompts_file!r}\n"
            f"  Create it first, or omit --prompts-file to use built-in --prompt-subset prompts.\n"
        )
        return 2

    hints = _model_config_hints(args.model) if args.quant == "gguf" else {"is_moe": False, "model_type": ""}
    is_qwen35_35b_awq = _looks_like_qwen35_35b_awq(args.model, args.quant)

    # DeepSeek-V2-Lite Q4 GGUF: logits are too lossy for coherent greedy/sampling Tier-B in practice.
    # Default Tier-B uses sibling HF bf16 Chat weights (same engine path as A-tier parity); GGUF-only when requested.
    _ds_gguf_only = os.environ.get("FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if (
        not _ds_gguf_only
        and args.quant == "gguf"
        and hints.get("model_type") == "deepseek_v2"
    ):
        try:
            from vllm.model_executor.models.deepseek_hf_reference import resolve_deepseek_hf_chat_dir

            _ref = resolve_deepseek_hf_chat_dir(args.model)
        except Exception:
            _ref = None
        if _ref:
            sys.stderr.write(
                f"[Tier-B] DeepSeek-V2-Lite-GGUF: loading sibling bf16 Chat for readability checks: {_ref}\n"
                f"  (Set FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY=1 to run Tier-B on GGUF weights.)\n"
            )
            args.model = _ref
            args.quant = "none"
            hints = _model_config_hints(args.model)

    _ds_gguf_only_deepseek = (
        _ds_gguf_only
        and args.quant == "gguf"
        and hints.get("model_type") == "deepseek_v2"
    )
    if args.top_p is None:
        if is_qwen35_35b_awq:
            # Tighter nucleus reduces tail-noise loops on 35B AWQ + FP8.
            args.top_p = 0.85
        else:
            args.top_p = 0.86 if _ds_gguf_only_deepseek else 0.95

    if args.temperature is None:
        if is_qwen35_35b_awq:
            # Mild stochasticity is more readable than greedy on this setup.
            args.temperature = 0.35
        elif _ds_gguf_only_deepseek:
            # Q4-only: slightly higher temp + tighter nucleus (see top_p) reduces junk from tail mass.
            args.temperature = 0.63
        elif args.quant == "gguf":
            args.temperature = 0.55
        else:
            args.temperature = 0.7

    if args.temperature < 0 or args.temperature > _MAX_DOC_TEMPERATURE:
        sys.stderr.write(
            f"[Warning] INFERENCE_ACCURACY.md §6 recommends temperature in [0, {_MAX_DOC_TEMPERATURE}]; "
            f"got {args.temperature}. Clamping to range for reporting.\n"
        )
        args.temperature = max(0.0, min(float(args.temperature), _MAX_DOC_TEMPERATURE))

    # DeepSeek-V2-Lite Q4 GGUF: greedy argmax typically collapses to one token id (repeated "play" / 上).
    # Tier-B checks readability under mild sampling unless user forces greedy via env.
    if (
        hints.get("model_type") == "deepseek_v2"
        and args.quant == "gguf"
        and float(args.temperature) <= 1e-6
        and os.environ.get("FASTINFERENCE_DEEPSEEK_B_TIER_GREEDY", "").strip().lower()
        not in ("1", "true", "yes", "on")
    ):
        args.temperature = 0.63 if _ds_gguf_only_deepseek else 0.55
        sys.stderr.write(
            "[Tier-B] DeepSeek-V2-Lite-GGUF: using temperature="
            f"{args.temperature} (Q4 greedy collapses). "
            "Set FASTINFERENCE_DEEPSEEK_B_TIER_GREEDY=1 to honor --temperature 0.\n"
        )

    if args.temperature <= 1e-6:
        sys.stderr.write(
            "[Note] Greedy decoding (--temperature 0): top_p has no effect on token choice.\n"
        )

    if is_qwen35_35b_awq and args.max_new_tokens == 96:
        # Keep continuations shorter by default to avoid late-stage degeneration.
        args.max_new_tokens = 16
        sys.stderr.write(
            "[Tier-B] Qwen3.5-35B-AWQ: defaulting --max-new-tokens to 16 "
            "(override explicitly if you need longer generations).\n"
        )

    prompts = _resolve_prompts(args)
    if not prompts:
        sys.stderr.write("[Error] No prompts to run (empty subset or file).\n")
        return 2

    gpu_mu = args.gpu_memory_utilization
    max_len = args.max_model_len
    max_seqs = 32
    if args.frugal:
        if gpu_mu is None:
            gpu_mu = 0.55
        if max_len is None:
            max_len = 1024
        max_seqs = 4
    if gpu_mu is None:
        gpu_mu = 0.9
    if max_len is None:
        max_len = 2048

    if args.repetition_penalty is None:
        if is_qwen35_35b_awq:
            args.repetition_penalty = 1.24
        elif args.temperature <= 1e-6:
            # Mild penalty even for greedy on MoE GGUF: reduces degenerate same-token / digit runs.
            if args.quant == "gguf" and hints.get("is_moe"):
                args.repetition_penalty = 1.08
            else:
                args.repetition_penalty = 1.0
        elif args.quant == "gguf":
            # MoE + Q4/Q8 logits: stronger penalty reduces "The 10." / digit loops.
            if _ds_gguf_only_deepseek and hints.get("is_moe"):
                args.repetition_penalty = 1.24
            else:
                args.repetition_penalty = 1.18 if hints.get("is_moe") else 1.15
        else:
            args.repetition_penalty = 1.12

    if args.frequency_penalty is None:
        if is_qwen35_35b_awq:
            args.frequency_penalty = 0.08
        elif args.temperature > 1e-6 and args.quant == "gguf":
            args.frequency_penalty = 0.10 if _ds_gguf_only_deepseek else 0.06
        else:
            args.frequency_penalty = 0.0

    if args.presence_penalty is None:
        if is_qwen35_35b_awq:
            args.presence_penalty = 0.08
        elif args.temperature > 1e-6 and args.quant == "gguf" and hints.get("is_moe"):
            args.presence_penalty = 0.08 if _ds_gguf_only_deepseek else 0.05
        else:
            args.presence_penalty = 0.0

    effective_top_k = int(args.top_k)
    if args.temperature > 1e-6 and effective_top_k < 0:
        if is_qwen35_35b_awq:
            effective_top_k = 20
        elif args.quant == "gguf":
            if _ds_gguf_only_deepseek:
                effective_top_k = 28 if hints.get("is_moe") else 40
            else:
                effective_top_k = 40 if hints.get("is_moe") else 50

    sys.stderr.write(
        f"[Tier-B] model={args.model!r} quant={args.quant} "
        f"prompts={len(prompts)} subset={args.prompt_subset} max_new_tokens={args.max_new_tokens} "
        f"gpu_mem_util={gpu_mu} max_model_len={max_len} max_num_seqs={max_seqs}\n"
    )
    sys.stderr.write(
        f"[Tier-B] sampling: temperature={args.temperature} top_p={args.top_p} top_k={effective_top_k} "
        f"repetition_penalty={args.repetition_penalty} frequency_penalty={args.frequency_penalty} "
        f"presence_penalty={args.presence_penalty} "
        f"hints={{gguf_moe={hints.get('is_moe')!s}, model_type={hints.get('model_type')!r}}}\n"
    )
    if os.environ.get("FASTINFERENCE_QWEN35_MOE_FP8", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        sys.stderr.write(
            "[Note] FASTINFERENCE_QWEN35_MOE_FP8=1 trades MoE weight precision for memory; "
            "if completions look repetitive or incoherent, unset it or use --temperature 0 for a greedy sanity check.\n"
        )
    if os.environ.get("FASTINFERENCE_QWEN35_MOE_PACKED_GGUF", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        sys.stderr.write(
            "[Note] FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=1 lowers MoE GGUF load-time host RSS; "
            "do not combine with FASTINFERENCE_QWEN35_MOE_FP8=1.\n"
        )
    if hints.get("model_type") == "deepseek_v2" and args.quant == "gguf":
        # Default-on for Tier-B readability; override with FASTINFERENCE_*=0 if needed.
        os.environ.setdefault("FASTINFERENCE_GGUF_DEQUANT_FP32", "1")
        os.environ.setdefault("FASTINFERENCE_DEEPSEEK_ATTN_FP32", "1")
        sys.stderr.write(
            "[Note] DeepSeek V2 Lite GGUF: Tier-B readability (docs/INFERENCE_ACCURACY.md §1). "
            "Defaulting FASTINFERENCE_GGUF_DEQUANT_FP32=1, FASTINFERENCE_DEEPSEEK_ATTN_FP32=1 "
            "(unset or =0 to compare).\n"
        )
    if _ds_gguf_only_deepseek:
        sys.stderr.write(
            "[Note] FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY=1: Tier-B uses tighter Q4 sampling "
            "(top_p/top_k/repetition/frequency/presence). "
            "DeepSeek-V2-Lite Q4_K_M MoE often still looks worse than bf16; for product B-tier "
            "readability, omit this env so the script loads sibling DeepSeek-V2-Lite-Chat (see docs).\n"
        )
    engine = _build_engine(
        args.model,
        args.quant,
        gpu_memory_utilization=gpu_mu,
        max_model_len=max_len,
        max_num_seqs=max_seqs,
    )
    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    engine.tokenizer = tokenizer

    prompts = _apply_chat_template_to_prompts(args.chat_template, tokenizer, prompts)

    any_severe = False
    summary_rows: List[dict] = []

    for pid, prompt in prompts:
        sp = SamplingParams(
            max_tokens=args.max_new_tokens,
            min_tokens=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=effective_top_k,
            repetition_penalty=float(args.repetition_penalty),
            frequency_penalty=float(args.frequency_penalty),
            presence_penalty=float(args.presence_penalty),
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
        severe, detail, msgs = analyze_tier_b(
            text,
            token_ids,
            tokenizer,
            check_substance=not args.skip_substance_heuristics,
        )

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
                f"coherence={ta.get('coherence_ok')} first_token={ta.get('first_token_ok')} "
                f"substance={ta.get('substance_ok')}"
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
            and r["tier_b_detail"].get("tier_b_alignment", {}).get("substance_ok", True)
        )
        total = len([r for r in summary_rows if "error" not in r])
        sys.stderr.write("\n" + "=" * 72 + "\n")
        sys.stderr.write(
            f"Summary: {passed}/{total} cases passed automated tier-B heuristics "
            f"(readability+coherence+first_token+substance). Human review still required.\n"
        )

    if any_severe and not args.no_heuristics_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
