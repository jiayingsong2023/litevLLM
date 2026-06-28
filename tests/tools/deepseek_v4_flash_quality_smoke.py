#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import json
import mmap
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import torch
from jinja2 import Environment
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

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
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
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
_DEFAULT_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain the concept of gravity in one sentence.",
    "Translate 'Good morning' into Chinese.",
    "Write a one-line Python function that returns the factorial of n.",
]
_TOKENIZER_TOKENS_KEY = "tokenizer.ggml.tokens"
_TOKENIZER_MERGES_KEY = "tokenizer.ggml.merges"
_TOKENIZER_BOS_KEY = "tokenizer.ggml.bos_token_id"
_TOKENIZER_EOS_KEY = "tokenizer.ggml.eos_token_id"
_TOKENIZER_ADD_BOS_KEY = "tokenizer.ggml.add_bos_token"
_TOKENIZER_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_MAX_TOKEN_REPEAT_RATIO = 0.75
_MAX_CHAR_REPEAT_RUN = 24
_MIN_PRINTABLE_RATIO = 0.80
_MIN_WORDS_FOR_REPEAT_CHECK = 4
_MAX_TOP_WORD_RATIO = 0.45


class GGUFTokenizerMetadata(NamedTuple):
    tokens: list[str]
    merges: list[str]
    bos_token_id: int | None
    eos_token_id: int | None
    add_bos_token: bool
    chat_template: str | None


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
        default=None,
        help="User prompt text. When given, only this single prompt is run.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="JSON list of prompt strings. Defaults to a small built-in suite.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Encode --prompt-text directly instead of applying GGUF chat template.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=str,
        default="",
        help=(
            "Comma-separated prompt token ids. "
            "Overrides --prompt-text for single-prompt mode."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-decode-tps", type=float, default=0.0)
    parser.add_argument("--max-total-elapsed-ms", type=float, default=0.0)
    parser.add_argument("--min-output-chars", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--dump-step-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--enable-profiler", action="store_true")
    return parser.parse_args(argv)


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    """Return the list of prompts to run.

    Precedence:
      1. ``--prompt-text`` for single-prompt mode (backwards compatibility).
      2. ``--prompts`` JSON list supplied by the caller.
      3. Built-in default suite.
    """
    if args.prompt_text is not None:
        return [args.prompt_text]
    if args.prompts is not None:
        parsed = json.loads(args.prompts)
        if not isinstance(parsed, list):
            raise ValueError("--prompts must be a JSON list of strings")
        return [str(p) for p in parsed]
    return list(_DEFAULT_PROMPTS)


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

    def bool(self) -> bool:
        return self.read(1) != b"\x00"

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
    return read_gguf_tokenizer_metadata(path).tokens


def read_gguf_tokenizer_metadata(path: Path) -> GGUFTokenizerMetadata:
    tokens: list[str] | None = None
    merges: list[str] = []
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    add_bos_token = False
    chat_template: str | None = None
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
                if key in (_TOKENIZER_TOKENS_KEY, _TOKENIZER_MERGES_KEY):
                    if value_type != GGUF_TYPE_ARRAY:
                        raise ValueError(f"{key} metadata must be an array")
                    element_type = cursor.u32()
                    element_count = cursor.u64()
                    if element_type != GGUF_TYPE_STRING:
                        raise ValueError(f"{key} array must contain strings")
                    values = [cursor.string() for _ in range(element_count)]
                    if key == _TOKENIZER_TOKENS_KEY:
                        tokens = values
                    else:
                        merges = values
                    continue
                if key == _TOKENIZER_BOS_KEY:
                    if value_type != 4:
                        raise ValueError("bos token id metadata must be uint32")
                    bos_token_id = cursor.u32()
                    continue
                if key == _TOKENIZER_EOS_KEY:
                    if value_type != 4:
                        raise ValueError("eos token id metadata must be uint32")
                    eos_token_id = cursor.u32()
                    continue
                if key == _TOKENIZER_ADD_BOS_KEY:
                    if value_type != 7:
                        raise ValueError("add_bos_token metadata must be bool")
                    add_bos_token = cursor.bool()
                    continue
                if key == _TOKENIZER_CHAT_TEMPLATE_KEY:
                    if value_type != GGUF_TYPE_STRING:
                        raise ValueError("chat template metadata must be a string")
                    chat_template = cursor.string()
                    continue
                _skip_metadata_value(cursor, value_type)
        finally:
            view.release()
            mm.close()
    if tokens is None:
        raise ValueError(f"missing GGUF metadata key: {_TOKENIZER_TOKENS_KEY}")
    return GGUFTokenizerMetadata(
        tokens=tokens,
        merges=merges,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        add_bos_token=add_bos_token,
        chat_template=chat_template,
    )


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
                f"{pos}: {prompt[pos : pos + 16]!r}"
            )
        token_ids.append(best_id)
        pos += len(best_piece)
    return token_ids


def _parse_bpe_merge(raw_merge: str) -> tuple[str, str]:
    left, sep, right = raw_merge.partition(" ")
    if not sep or not left or not right:
        raise ValueError(f"invalid GGUF BPE merge entry: {raw_merge!r}")
    return left, right


def build_bpe_prompt_tokenizer(
    *,
    gguf_tokens: list[str],
    gguf_merges: list[str],
    special_tokens: list[str] | None = None,
    unk_token: str | None = None,
) -> Tokenizer:
    vocab = {piece: token_id for token_id, piece in enumerate(gguf_tokens)}
    if unk_token is not None and unk_token not in vocab:
        unk_token = None
    tokenizer = Tokenizer(
        BPE(
            vocab=vocab,
            merges=[_parse_bpe_merge(merge) for merge in gguf_merges],
            unk_token=unk_token,
        )
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    if special_tokens:
        tokenizer.add_special_tokens(
            [token for token in special_tokens if token in vocab]
        )
    return tokenizer


def encode_prompt_text_bpe(prompt: str, *, tokenizer: Tokenizer) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(prompt).ids]


def special_tokens_from_vocab(gguf_tokens: list[str]) -> list[str]:
    return [
        token
        for token in gguf_tokens
        if (
            (token.startswith("<") and token.endswith(">"))
            or (token.startswith("｜") and token.endswith("｜"))
        )
    ]


def encode_prompt_text_with_metadata(
    prompt: str,
    *,
    tokenizer_metadata: GGUFTokenizerMetadata,
) -> list[int]:
    if tokenizer_metadata.merges:
        tokenizer = build_bpe_prompt_tokenizer(
            gguf_tokens=tokenizer_metadata.tokens,
            gguf_merges=tokenizer_metadata.merges,
            special_tokens=special_tokens_from_vocab(tokenizer_metadata.tokens),
        )
        token_ids = encode_prompt_text_bpe(prompt, tokenizer=tokenizer)
    else:
        token_ids = encode_prompt_text(
            prompt,
            gguf_tokens=tokenizer_metadata.tokens,
        )
    if (
        tokenizer_metadata.add_bos_token
        and tokenizer_metadata.bos_token_id is not None
        and (not token_ids or token_ids[0] != tokenizer_metadata.bos_token_id)
    ):
        token_ids.insert(0, tokenizer_metadata.bos_token_id)
    return token_ids


def render_chat_prompt(
    prompt: str,
    *,
    chat_template: str,
    bos_token: str,
    eos_token: str,
) -> str:
    environment = Environment(autoescape=False)
    environment.filters["from_json"] = json.loads
    template = environment.from_string(chat_template)
    return str(
        template.render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            thinking=False,
            enable_thinking=False,
            bos_token=bos_token,
            eos_token=eos_token,
        )
    )


def prompt_text_for_encoding(
    prompt: str,
    *,
    tokenizer_metadata: GGUFTokenizerMetadata,
    raw_prompt: bool,
) -> str:
    if raw_prompt or tokenizer_metadata.chat_template is None:
        return prompt
    bos_token = (
        tokenizer_metadata.tokens[tokenizer_metadata.bos_token_id]
        if tokenizer_metadata.bos_token_id is not None
        else ""
    )
    eos_token = (
        tokenizer_metadata.tokens[tokenizer_metadata.eos_token_id]
        if tokenizer_metadata.eos_token_id is not None
        else ""
    )
    return render_chat_prompt(
        prompt,
        chat_template=tokenizer_metadata.chat_template,
        bos_token=bos_token,
        eos_token=eos_token,
    )


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


def topk_records(
    torch_logits: torch.Tensor | list[float],
    *,
    gguf_tokens: list[str],
    k: int,
) -> list[dict[str, object]]:
    logits = (
        torch_logits.detach().to(device="cpu", dtype=torch.float32)
        if isinstance(torch_logits, torch.Tensor)
        else torch.tensor(torch_logits, dtype=torch.float32)
    )
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    k = max(1, min(int(k), int(logits.numel())))
    values, indices = torch.topk(logits, k=k)
    records: list[dict[str, object]] = []
    for token_id, logit in zip(indices.tolist(), values.tolist(), strict=True):
        records.append(
            {
                "token_id": int(token_id),
                "text": decode_generated_tokens(
                    [int(token_id)],
                    gguf_tokens=gguf_tokens,
                ),
                "logit": float(logit),
            }
        )
    return records


def logits_stats(logits: torch.Tensor) -> dict[str, float | bool]:
    logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    finite = torch.isfinite(logits_cpu)
    finite_logits = logits_cpu[finite]
    if finite_logits.numel() == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "has_nan": bool(torch.isnan(logits_cpu).any().item()),
            "has_inf": bool(torch.isinf(logits_cpu).any().item()),
        }
    return {
        "min": float(finite_logits.min().item()),
        "max": float(finite_logits.max().item()),
        "mean": float(finite_logits.mean().item()),
        "std": float(finite_logits.std(unbiased=False).item()),
        "has_nan": bool(torch.isnan(logits_cpu).any().item()),
        "has_inf": bool(torch.isinf(logits_cpu).any().item()),
    }


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
    eos_token_id: int | None = 1,
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
        if eos_token_id is not None and eos_token_id in generated_token_ids[:-1]:
            reasons.append("tokens_after_eos")
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


def _build_ready_backend(args: argparse.Namespace) -> DeepSeekV4FlashGPUBackend:
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
    gguf_tokens: list[str],
    eos_token_id: int | None,
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
            gpu_backend=_build_ready_backend(args),
        )
        if bool(getattr(args, "enable_profiler", False)):
            model.enable_deepseek_profile()
        request_state = DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=args.context_length,
                hidden_size=model.shape.hidden_size,
                batch_size=1,
                kv_width=model.shape.head_dim,
                device=torch.device("cuda"),
            )
        )
        output_ids = list(prompt_token_ids)
        step_records: list[dict[str, object]] = []
        next_token_tensor: torch.Tensor | None = None
        for token_idx, token_id in enumerate(prompt_token_ids):
            logits = model._forward_kernel_token_step(
                token_id=int(token_id),
                state=request_state,
                token_idx=token_idx,
                device=torch.device("cuda"),
            )
            next_token_tensor = torch.argmax(logits, dim=-1).to(torch.long).reshape(())
            step_records.append(
                {
                    "phase": "prefill",
                    "token_idx": token_idx,
                    "input_token_id": int(token_id),
                    "input_text": decode_generated_tokens(
                        [int(token_id)], gguf_tokens=gguf_tokens
                    ),
                    "selected_token_id": int(next_token_tensor.item()),
                    "selected_text": decode_generated_tokens(
                        [int(next_token_tensor.item())],
                        gguf_tokens=gguf_tokens,
                    ),
                    "topk": topk_records(
                        logits,
                        gguf_tokens=gguf_tokens,
                        k=args.top_k,
                    ),
                    "logits": logits_stats(logits),
                }
            )
        if next_token_tensor is None:
            raise RuntimeError("DeepSeek quality smoke requires prompt tokens")
        for generated_idx in range(args.max_tokens):
            selected_id = int(next_token_tensor.item())
            output_ids.append(selected_id)
            if selected_id == eos_token_id or generated_idx + 1 >= args.max_tokens:
                break
            logits = model._forward_kernel_token_step(
                token_id=selected_id,
                state=request_state,
                token_idx=len(output_ids) - 1,
                device=torch.device("cuda"),
            )
            next_token_tensor = torch.argmax(logits, dim=-1).to(torch.long).reshape(())
            step_records.append(
                {
                    "phase": "decode",
                    "token_idx": len(output_ids) - 1,
                    "input_token_id": selected_id,
                    "input_text": decode_generated_tokens(
                        [selected_id], gguf_tokens=gguf_tokens
                    ),
                    "selected_token_id": int(next_token_tensor.item()),
                    "selected_text": decode_generated_tokens(
                        [int(next_token_tensor.item())],
                        gguf_tokens=gguf_tokens,
                    ),
                    "topk": topk_records(
                        logits,
                        gguf_tokens=gguf_tokens,
                        k=args.top_k,
                    ),
                    "logits": logits_stats(logits),
                }
            )
        torch.cuda.synchronize()
        return {
            "output_token_ids": output_ids,
            "step_records": step_records,
            "gpu_backend": model.gpu_backend.stats(),
            "gpu_staging": model.gpu_staging_memory_stats(),
            "profiler": model.deepseek_profile(),
        }


def _args_with_max_tokens(
    args: argparse.Namespace,
    max_tokens: int,
) -> argparse.Namespace:
    updated = vars(args).copy()
    updated["max_tokens"] = max_tokens
    return argparse.Namespace(**updated)


def _run_single_prompt(
    args: argparse.Namespace,
    *,
    prompt_text: str,
    tokenizer_metadata: GGUFTokenizerMetadata,
    gguf_tokens: list[str],
) -> dict[str, object]:
    """Run the quality smoke for one prompt and return a per-case payload."""
    encoded_prompt_text = prompt_text_for_encoding(
        prompt_text,
        tokenizer_metadata=tokenizer_metadata,
        raw_prompt=bool(args.raw_prompt),
    )
    prompt_token_ids = parse_prompt_token_ids(args.prompt_token_ids)
    if not prompt_token_ids:
        prompt_token_ids = encode_prompt_text_with_metadata(
            encoded_prompt_text,
            tokenizer_metadata=tokenizer_metadata,
        )
    repeat = max(1, int(getattr(args, "repeat", 1)))
    min_decode_tps = float(getattr(args, "min_decode_tps", 0.0))
    max_total_elapsed_ms = float(getattr(args, "max_total_elapsed_ms", 0.0))

    measured_payloads: list[dict[str, Any]] = []
    performance_values: list[dict[str, float | int | str]] = []
    for _ in range(repeat):
        generation_start = perf_counter()
        measured_payload = _run_direct_generate(
            args,
            prompt_token_ids=prompt_token_ids,
            gguf_tokens=gguf_tokens,
            eos_token_id=tokenizer_metadata.eos_token_id,
        )
        total_elapsed_ms = (perf_counter() - generation_start) * 1000.0
        measured_payloads.append(measured_payload)
        output_ids_for_metrics = [
            int(token_id) for token_id in measured_payload.get("output_token_ids", [])
        ]
        generated_ids_for_metrics = output_ids_for_metrics[len(prompt_token_ids) :]
        performance_values.append(
            _performance_metrics(
                generated_ids=generated_ids_for_metrics,
                total_elapsed_ms=total_elapsed_ms,
            )
        )
    smoke_payload = measured_payloads[0]
    performance = performance_values[0]
    output_ids = [
        int(token_id) for token_id in smoke_payload.get("output_token_ids", [])
    ]
    generated_ids = output_ids[len(prompt_token_ids) :]
    decode_tps_values = [
        float(metrics["decode_tokens_per_second"]) for metrics in performance_values
    ]
    performance_summary = {
        "repeat": repeat,
        "decode_tps_values": decode_tps_values,
        "decode_tps_min": min(decode_tps_values),
        "decode_tps_median": float(statistics.median(decode_tps_values)),
        "decode_tps_max": max(decode_tps_values),
    }
    performance_gates = _performance_gate_verdict(
        decode_tps_min=float(performance_summary["decode_tps_min"]),
        total_elapsed_ms=float(performance["total_elapsed_ms"]),
        min_decode_tps=min_decode_tps,
        max_total_elapsed_ms=max_total_elapsed_ms,
    )
    text = decode_generated_tokens(generated_ids, gguf_tokens=gguf_tokens)
    verdict = evaluate_readability(
        text=text,
        generated_token_ids=generated_ids,
        min_output_chars=args.min_output_chars,
        eos_token_id=tokenizer_metadata.eos_token_id,
    )
    return {
        "prompt_text": prompt_text,
        "encoded_prompt_text": encoded_prompt_text,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_ids,
        "generated_token_ids": generated_ids,
        "readability": verdict,
        "performance": performance,
        "performance_summary": performance_summary,
        "performance_gates": performance_gates,
        "step_records": smoke_payload.get("step_records", []),
        "gpu_backend": smoke_payload.get("gpu_backend", {}),
        "gpu_staging": smoke_payload.get("gpu_staging", {}),
    }


def _performance_metrics(
    *,
    generated_ids: list[int],
    total_elapsed_ms: float,
) -> dict[str, float | int | str]:
    decode_tokens_per_second = (
        len(generated_ids) / (total_elapsed_ms / 1000.0)
        if total_elapsed_ms > 0.0
        else 0.0
    )
    return {
        "generated_tokens": len(generated_ids),
        "metric_scope": "direct_generation_total",
        "total_elapsed_ms": total_elapsed_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
    }


def _performance_gate_verdict(
    *,
    decode_tps_min: float,
    total_elapsed_ms: float,
    min_decode_tps: float,
    max_total_elapsed_ms: float,
) -> dict[str, object]:
    reasons: list[str] = []
    if min_decode_tps > 0.0 and decode_tps_min < min_decode_tps:
        reasons.append("decode_tps_below_min")
    if max_total_elapsed_ms > 0.0 and total_elapsed_ms > max_total_elapsed_ms:
        reasons.append("total_elapsed_above_max")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "min_decode_tps": min_decode_tps,
        "max_total_elapsed_ms": max_total_elapsed_ms,
    }


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"DeepSeek V4 Flash GGUF file not found: {args.model}")
    tokenizer_metadata = read_gguf_tokenizer_metadata(args.model)
    gguf_tokens = tokenizer_metadata.tokens
    prompts = _resolve_prompts(args)
    if not prompts:
        raise SystemExit("No prompts to run")

    warmup_tokens = int(getattr(args, "warmup_tokens", 0))
    if warmup_tokens > 0:
        warmup_prompt = prompts[0]
        warmup_encoded = prompt_text_for_encoding(
            warmup_prompt,
            tokenizer_metadata=tokenizer_metadata,
            raw_prompt=bool(args.raw_prompt),
        )
        warmup_token_ids = encode_prompt_text_with_metadata(
            warmup_encoded,
            tokenizer_metadata=tokenizer_metadata,
        )
        _run_direct_generate(
            _args_with_max_tokens(args, warmup_tokens),
            prompt_token_ids=warmup_token_ids,
            gguf_tokens=gguf_tokens,
            eos_token_id=tokenizer_metadata.eos_token_id,
        )

    cases: list[dict[str, object]] = []
    for prompt_text in prompts:
        case_payload = _run_single_prompt(
            args,
            prompt_text=prompt_text,
            tokenizer_metadata=tokenizer_metadata,
            gguf_tokens=gguf_tokens,
        )
        cases.append(case_payload)
        if "profiler" in case_payload.get("gpu_backend", {}):
            pass

    readability_passed = all(bool(c["readability"].get("passed")) for c in cases)
    performance_gates_passed = all(
        bool(c["performance_gates"].get("passed")) for c in cases
    )
    overall_passed = readability_passed and performance_gates_passed

    payload: dict[str, object] = {
        "model": str(args.model),
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "raw_prompt": bool(args.raw_prompt),
        "case_count": len(cases),
        "cases": cases,
        "readability_passed": readability_passed,
        "performance_gates_passed": performance_gates_passed,
        "overall_passed": overall_passed,
    }
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    if args.dump_step_json is not None:
        _write_json(
            args.dump_step_json,
            {
                "prompts": [c["prompt_text"] for c in cases],
                "cases": [
                    {
                        "prompt_token_ids": c["prompt_token_ids"],
                        "generated_token_ids": c["generated_token_ids"],
                        "step_records": c["step_records"],
                    }
                    for c in cases
                ],
            },
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
