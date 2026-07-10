#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
P1 tokenizer gate for Gemma4 speculative decoding.

Loads the target and draft tokenizers (no model weights) and verifies that
both tokenizers agree on vocabulary, added tokens, special token IDs, prompt
encoding, and chat-template rendering.  If every hard check passes, P1 may
proceed; otherwise it stops.
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_PROMPTS_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "gemma4_speculative_prompts.json"
)


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


def _vocab_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()  # dict token -> id
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode("utf-8")).hexdigest()


def _added_tokens(tokenizer: Any) -> dict[str, int]:
    added = tokenizer.added_tokens_encoder  # type: ignore[attr-defined]
    return {str(k): int(v) for k, v in (added or {}).items()}


def _expand_to_token_ids(tokenizer: Any, seed: str, target_len: int) -> list[int]:
    tokens = tokenizer.encode(seed)
    if len(tokens) >= target_len:
        return tokens[:target_len]
    repeated = (seed + " ") * ((target_len // len(tokens)) + 2)
    tokens = tokenizer.encode(repeated)
    return tokens[:target_len]


def check_vocab_size(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    """Compare tokenizer.vocab_size values."""
    match = target.vocab_size == draft.vocab_size
    details = {"target": target.vocab_size, "draft": draft.vocab_size}
    return match, details


def check_full_vocab(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    """Compare SHA-256 hashes of the full vocabulary mapping."""
    target_hash = _vocab_hash(target)
    draft_hash = _vocab_hash(draft)
    return target_hash == draft_hash, {
        "target_hash": target_hash,
        "draft_hash": draft_hash,
    }


def check_added_tokens(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    """Compare added-tokens encoders."""
    target_added = _added_tokens(target)
    draft_added = _added_tokens(draft)
    return target_added == draft_added, {
        "target": target_added,
        "draft": draft_added,
    }


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


def check_chat_template_token_ids(
    target: Any, draft: Any, messages: list[dict[str, str]]
) -> tuple[bool | None, dict[str, Any]]:
    """Compare token IDs produced by applying the chat template."""
    target_template = getattr(target, "chat_template", None)
    draft_template = getattr(draft, "chat_template", None)
    if target_template is None or draft_template is None:
        details = {
            "target_has_template": target_template is not None,
            "draft_has_template": draft_template is not None,
            "target_ids": None,
            "draft_ids": None,
        }
        return None, details

    target_text = target.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    draft_text = draft.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    target_ids = target.encode(target_text, add_special_tokens=False)
    draft_ids = draft.encode(draft_text, add_special_tokens=False)
    match = target_ids == draft_ids
    details = {
        "target_text": target_text,
        "draft_text": draft_text,
        "target_ids": target_ids,
        "draft_ids": draft_ids,
    }
    return match, details


def build_report(
    target_model: str,
    draft_model: str,
    target: Any,
    draft: Any,
    prompts: list[str],
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run all tokenizer compatibility checks and assemble the report."""
    if messages is None:
        messages = DEFAULT_CHAT_MESSAGES

    vocab_match, vocab_details = check_vocab_size(target, draft)
    full_vocab_match, full_vocab_details = check_full_vocab(target, draft)
    added_match, added_details = check_added_tokens(target, draft)
    special_match, special_details = check_special_tokens(target, draft)
    encode_match, encode_details = check_encode(target, draft, prompts)
    chat_match, chat_details = check_chat_template(target, draft, messages)
    chat_token_ids_match, chat_token_ids_details = check_chat_template_token_ids(
        target, draft, messages
    )

    passed = full_vocab_match and added_match and special_match and encode_match

    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "vocab_size_match": vocab_match,
        "full_vocab_match": full_vocab_match,
        "added_tokens_match": added_match,
        "special_tokens_match": special_match,
        "encode_match": encode_match,
        "chat_template_match": chat_match,
        "chat_template_token_ids_match": chat_token_ids_match,
        "passed": passed,
        "details": {
            "vocab_size": vocab_details,
            "full_vocab": full_vocab_details,
            "added_tokens": added_details,
            "special_tokens": special_details,
            "encode": encode_details,
            "chat_template": chat_details,
            "chat_template_token_ids": chat_token_ids_details,
        },
    }


def _load_prompts_from_fixture(tokenizer: Any, fixture_path: Path) -> list[str]:
    """Load seed prompts from a fixture and expand them to exact token lengths."""
    data = json.loads(fixture_path.read_text())
    expanded_prompts: list[str] = []
    for item in data["prompts"]:
        token_ids = _expand_to_token_ids(tokenizer, item["text"], item["context_len"])
        actual_len = len(token_ids)
        if actual_len != item["context_len"]:
            raise ValueError(
                f"Prompt {item['id']!r}: expected {item['context_len']} tokens, "
                f"got {actual_len}"
            )
        expanded_prompts.append(tokenizer.decode(token_ids))
    return expanded_prompts


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
        "--prompts-fixture",
        type=str,
        default=str(DEFAULT_PROMPTS_FIXTURE),
        help="Path to JSON prompts fixture",
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

    target = _load_tokenizer(args.target_model)

    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        fixture_path = Path(args.prompts_fixture)
        if fixture_path.exists():
            prompts = _load_prompts_from_fixture(target, fixture_path)
        else:
            prompts = DEFAULT_PROMPTS

    draft = _load_tokenizer(args.draft_model)

    report = build_report(args.target_model, args.draft_model, target, draft, prompts)

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.json_out:
        Path(args.json_out).write_text(report_json)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
