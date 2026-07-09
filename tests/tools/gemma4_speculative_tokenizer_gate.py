#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
P1 tokenizer gate for Gemma4 speculative decoding.

Loads the target and draft tokenizers (no model weights) and verifies that
both tokenizers agree on vocabulary size, special token IDs, prompt encoding,
and chat-template rendering.  If every check passes, P1 may proceed; otherwise
it stops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vllm.model_executor.model_loader import get_tokenizer

DEFAULT_TARGET_MODEL = "models/gemma-4-26B-A4B-it-AWQ-4bit"
DEFAULT_DRAFT_MODEL = "models/gemma-4-E2B-it-AWQ-INT4"

DEFAULT_PROMPTS: list[str] = [
    "The capital of France is",
    "法国的首都是",
    "A quick brown fox jumps over the lazy dog.",
    "Explain quantum mechanics in one sentence.",
]

DEFAULT_CHAT_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Hello, how are you?"},
]


def _load_tokenizer(model_path: str) -> Any:
    """Load only the tokenizer files for a local model directory."""
    return get_tokenizer(model_path, trust_remote_code=True)


def _normalize_token_ids(value: Any) -> list[int]:
    """Convert a single int, list of ints, or None into a sorted list of ints."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return sorted(int(x) for x in value)


def _special_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    return {
        "bos_token_id": _normalize_token_ids(getattr(tokenizer, "bos_token_id", None)),
        "eos_token_id": _normalize_token_ids(getattr(tokenizer, "eos_token_id", None)),
        "pad_token_id": _normalize_token_ids(getattr(tokenizer, "pad_token_id", None)),
    }


def check_vocab_size(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    """Compare tokenizer.vocab_size values."""
    match = target.vocab_size == draft.vocab_size
    details = {"target": target.vocab_size, "draft": draft.vocab_size}
    return match, details


def check_special_tokens(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    """Compare normalized bos/eos/pad token IDs."""
    target_ids = _special_token_ids(target)
    draft_ids = _special_token_ids(draft)
    match = target_ids == draft_ids
    details = {"target": target_ids, "draft": draft_ids}
    return match, details


def check_encode(
    target: Any, draft: Any, prompts: list[str]
) -> tuple[bool, dict[str, Any]]:
    """Compare encoded token IDs for every prompt, ignoring special tokens."""
    per_prompt: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for prompt in prompts:
        target_ids = target.encode(prompt, add_special_tokens=False)
        draft_ids = draft.encode(prompt, add_special_tokens=False)
        prompt_match = target_ids == draft_ids
        per_prompt.append(
            {
                "prompt": prompt,
                "target_ids": target_ids,
                "draft_ids": draft_ids,
                "match": prompt_match,
            }
        )
        if not prompt_match:
            mismatches.append(prompt)
    match = not mismatches
    details = {"prompts": per_prompt, "mismatches": mismatches}
    return match, details


def check_chat_template(
    target: Any,
    draft: Any,
    messages: list[dict[str, str]] | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Compare chat-template rendering if both tokenizers expose one."""
    if messages is None:
        messages = DEFAULT_CHAT_MESSAGES
    target_template = getattr(target, "chat_template", None)
    draft_template = getattr(draft, "chat_template", None)
    if target_template is None or draft_template is None:
        details = {
            "target_has_template": target_template is not None,
            "draft_has_template": draft_template is not None,
            "target_output": None,
            "draft_output": None,
        }
        return None, details

    target_output = target.apply_chat_template(messages, tokenize=False)
    draft_output = draft.apply_chat_template(messages, tokenize=False)
    match = target_output == draft_output
    details = {
        "target_output": target_output,
        "draft_output": draft_output,
    }
    return match, details


def build_report(
    target_model: str,
    draft_model: str,
    target: Any,
    draft: Any,
    prompts: list[str],
) -> dict[str, Any]:
    """Run all tokenizer compatibility checks and assemble the report."""
    vocab_match, vocab_details = check_vocab_size(target, draft)
    special_match, special_details = check_special_tokens(target, draft)
    encode_match, encode_details = check_encode(target, draft, prompts)
    chat_match, chat_details = check_chat_template(target, draft)

    passed = (
        vocab_match
        and special_match
        and encode_match
        and (chat_match is None or chat_match)
    )

    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "vocab_size_match": vocab_match,
        "special_tokens_match": special_match,
        "encode_match": encode_match,
        "chat_template_match": chat_match,
        "passed": passed,
        "details": {
            "vocab_size": vocab_details,
            "special_tokens": special_details,
            "encode": encode_details,
            "chat_template": chat_details,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P1 tokenizer gate for Gemma4 speculative decoding"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=DEFAULT_TARGET_MODEL,
        help="Path or name of the target (large) model tokenizer",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help="Path or name of the draft (small) model tokenizer",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated list of test prompts",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Path to write the JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    prompts = (
        [p.strip() for p in args.prompts.split(",") if p.strip()]
        if args.prompts
        else DEFAULT_PROMPTS
    )

    target = _load_tokenizer(args.target_model)
    draft = _load_tokenizer(args.draft_model)

    report = build_report(args.target_model, args.draft_model, target, draft, prompts)

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.json_out:
        Path(args.json_out).write_text(report_json)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
