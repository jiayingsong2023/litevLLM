#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import json
import mmap
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUF_MAGIC,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_STRING,
    GGUF_VERSION,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

_DEFAULT_GGUF = (
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)
_TOKENIZER_TOKENS_KEY = "tokenizer.ggml.tokens"
_MAX_TOKEN_REPEAT_RATIO = 0.75
_MAX_CHAR_REPEAT_RUN = 24
_MIN_PRINTABLE_RATIO = 0.80
_MIN_WORDS_FOR_REPEAT_CHECK = 4
_MAX_TOP_WORD_RATIO = 0.45


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek V4 Flash GGUF direct GPU quality smoke."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(_DEFAULT_GGUF),
        help="Path to the DeepSeek V4 Flash GGUF file.",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="What is the capital of France?",
        help="Prompt text encoded with a simple GGUF token-piece longest match.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=str,
        default="",
        help="Comma-separated prompt token ids. Overrides --prompt-text.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--min-output-chars", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


class _Cursor:
    def __init__(self, data: memoryview) -> None:
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated GGUF")
        out = self.data[self.pos : self.pos + n].tobytes()
        self.pos += n
        return out

    def u32(self) -> int:
        return int.from_bytes(self.read(4), "little")

    def u64(self) -> int:
        return int.from_bytes(self.read(8), "little")

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def skip(self, n: int) -> None:
        if self.pos + n > len(self.data):
            raise ValueError("truncated GGUF")
        self.pos += n


def _skip_metadata_value(cursor: _Cursor, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        cursor.skip(cursor.u64())
        return
    if value_type == GGUF_TYPE_ARRAY:
        element_type = cursor.u32()
        element_count = cursor.u64()
        if element_type == GGUF_TYPE_STRING:
            for _ in range(element_count):
                cursor.skip(cursor.u64())
            return
        primitive_sizes = {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 1,
            10: 8,
            11: 8,
            12: 8,
        }
        size = primitive_sizes.get(element_type)
        if size is None:
            raise ValueError(f"unsupported GGUF array type: {element_type}")
        cursor.skip(size * element_count)
        return
    primitive_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    size = primitive_sizes.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")
    cursor.skip(size)


def read_gguf_token_strings(path: Path) -> list[str]:
    with path.open("rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mm)
        try:
            cursor = _Cursor(view)
            magic = cursor.u32()
            if magic != GGUF_MAGIC:
                raise ValueError(f"invalid GGUF magic: 0x{magic:08x}")
            version = cursor.u32()
            if version != GGUF_VERSION:
                raise ValueError(f"unsupported GGUF version: {version}")
            _tensor_count = cursor.u64()
            metadata_count = cursor.u64()
            for _ in range(metadata_count):
                key = cursor.string()
                value_type = cursor.u32()
                if key != _TOKENIZER_TOKENS_KEY:
                    _skip_metadata_value(cursor, value_type)
                    continue
                if value_type != GGUF_TYPE_ARRAY:
                    raise ValueError("tokenizer tokens metadata must be an array")
                element_type = cursor.u32()
                element_count = cursor.u64()
                if element_type != GGUF_TYPE_STRING:
                    raise ValueError("tokenizer token array must contain strings")
                return [cursor.string() for _ in range(element_count)]
        finally:
            view.release()
            mm.close()
    raise ValueError(f"missing GGUF metadata key: {_TOKENIZER_TOKENS_KEY}")


def _normalize_token_piece(piece: str) -> str:
    with contextlib.suppress(UnicodeError):
        piece = piece.encode("latin1").decode("utf-8")
    if piece.startswith("<") and piece.endswith(">"):
        return ""
    return piece.replace("▁", " ").replace("Ġ", " ")


def encode_prompt_text(prompt: str, *, gguf_tokens: list[str]) -> list[int]:
    normalized_to_id: dict[str, int] = {}
    for token_id, piece in enumerate(gguf_tokens):
        normalized = _normalize_token_piece(piece)
        if normalized and normalized not in normalized_to_id:
            normalized_to_id[normalized] = token_id
    token_ids: list[int] = []
    pos = 0
    while pos < len(prompt):
        best_piece = ""
        best_id: int | None = None
        for piece, token_id in normalized_to_id.items():
            if len(piece) <= len(best_piece):
                continue
            if prompt.startswith(piece, pos):
                best_piece = piece
                best_id = token_id
        if best_id is None:
            raise ValueError(
                "could not encode prompt at byte offset "
                f"{pos}: {prompt[pos:pos + 16]!r}"
            )
        token_ids.append(best_id)
        pos += len(best_piece)
    return token_ids


def parse_prompt_token_ids(raw: str) -> list[int]:
    token_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        token_ids.append(int(item))
    return token_ids


def decode_generated_tokens(
    token_ids: list[int],
    *,
    gguf_tokens: list[str],
) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(gguf_tokens):
            continue
        pieces.append(_normalize_token_piece(gguf_tokens[token_id]))
    return "".join(pieces).strip()


def _max_consecutive_char_run(text: str) -> int:
    best = 0
    current = 0
    prev = ""
    for char in text:
        if char == prev:
            current += 1
        else:
            current = 1
            prev = char
        best = max(best, current)
    return best


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = 0
    for char in text:
        category = unicodedata.category(char)
        if not category.startswith("C") or char in "\n\t":
            printable += 1
    return printable / len(text)


def _word_repeat_ratio(text: str) -> float:
    words = [word.lower() for word in re.findall(r"[A-Za-z]{2,}", text)]
    if len(words) < _MIN_WORDS_FOR_REPEAT_CHECK:
        return 0.0
    counts = Counter(words)
    stem_counts = Counter(word[:3] for word in words if len(word) >= 3)
    exact_ratio = max(counts.values()) / len(words)
    stem_ratio = max(stem_counts.values()) / len(words) if stem_counts else 0.0
    return max(exact_ratio, stem_ratio)


def evaluate_readability(
    *,
    text: str,
    generated_token_ids: list[int],
    min_output_chars: int,
) -> dict[str, object]:
    reasons: list[str] = []
    stripped = text.strip()
    if len(stripped) < min_output_chars:
        reasons.append("too_short")
    if _printable_ratio(stripped) < _MIN_PRINTABLE_RATIO:
        reasons.append("low_printable_ratio")
    if _max_consecutive_char_run(stripped) > _MAX_CHAR_REPEAT_RUN:
        reasons.append("char_repeat")
    if _word_repeat_ratio(stripped) > _MAX_TOP_WORD_RATIO:
        reasons.append("word_repeat")
    if generated_token_ids:
        counts = Counter(generated_token_ids)
        top_ratio = max(counts.values()) / len(generated_token_ids)
        if top_ratio > _MAX_TOKEN_REPEAT_RATIO:
            reasons.append("token_repeat")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "text": stripped,
        "generated_token_ids": generated_token_ids,
        "char_count": len(stripped),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_ready_backend() -> DeepSeekV4FlashGPUBackend:
    return DeepSeekV4FlashGPUBackend(
        capabilities=DeepSeekV4FlashGPUCapabilities(
            q8_linear=True,
            attention=True,
            compressed_attention=True,
            cache_update=True,
            moe=True,
            output=True,
        )
    )


def _run_direct_generate(
    args: argparse.Namespace,
    *,
    prompt_token_ids: list[int],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm torch backend is not available")
    policy = DeepSeekV4FlashMemoryPolicy()
    with open_deepseek_v4_flash_weight_store(args.model) as store:
        budget = policy.estimate_runtime_budget(
            args.context_length,
            model_mmap_bytes=store.diagnostics.file_size_bytes,
        )
        policy.validate_runtime_budget(budget)
        model = DeepSeekV4FlashForCausalLM(
            weight_store=store,
            runtime_budget=budget,
            gpu_backend=_build_ready_backend(),
        )
        input_ids = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device="cuda",
        )
        output_ids = model.generate_greedy_kernel(
            input_ids,
            max_tokens=args.max_tokens,
        )
        torch.cuda.synchronize()
        return {
            "output_token_ids": [
                int(token_id) for token_id in output_ids.detach().cpu().tolist()
            ],
            "gpu_backend": model.gpu_backend.stats(),
            "gpu_staging": model.gpu_staging_memory_stats(),
        }


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"DeepSeek V4 Flash GGUF file not found: {args.model}")
    gguf_tokens = read_gguf_token_strings(args.model)
    prompt_token_ids = parse_prompt_token_ids(args.prompt_token_ids)
    if not prompt_token_ids:
        prompt_token_ids = encode_prompt_text(args.prompt_text, gguf_tokens=gguf_tokens)
    smoke_payload = _run_direct_generate(args, prompt_token_ids=prompt_token_ids)
    output_ids = [
        int(token_id) for token_id in smoke_payload.get("output_token_ids", [])
    ]
    generated_ids = output_ids[len(prompt_token_ids) :]
    text = decode_generated_tokens(generated_ids, gguf_tokens=gguf_tokens)
    verdict = evaluate_readability(
        text=text,
        generated_token_ids=generated_ids,
        min_output_chars=args.min_output_chars,
    )
    payload: dict[str, object] = {
        "model": str(args.model),
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "prompt_text": args.prompt_text,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_ids,
        "generated_token_ids": generated_ids,
        "readability": verdict,
        "gpu_backend": smoke_payload.get("gpu_backend", {}),
    }
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(verdict["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
