# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
import xgrammar as xgr
from jsonschema import ValidationError
from jsonschema import validate as validate_json_schema

from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


class LiteStructuredOutputConstraint:
    def apply(self, logits: torch.Tensor, request: RequestState) -> torch.Tensor:
        return logits

    def on_token(self, request: RequestState, token_id: int) -> bool:
        del request, token_id
        return True

    def should_finish(self, request: RequestState) -> bool:
        del request
        return False


def _escape_regex_choice(text: str) -> str:
    return re.escape(text)


def _choice_as_regex(choices: list[str]) -> str:
    return "^(?:" + "|".join(_escape_regex_choice(choice) for choice in choices) + ")$"


def _estimate_vocab_size(tokenizer: Any) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict) and vocab:
            return max(int(v) for v in vocab.values()) + 1
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id) + 1
    raise ValueError("unable to infer tokenizer vocab size for structured outputs")


@dataclass
class XgrammarStructuredOutputConstraint(LiteStructuredOutputConstraint):
    tokenizer: Any
    grammar_type: str
    grammar_spec: str
    vocab_size: int
    finish_mode: str = "matcher"
    choices: list[str] | None = None
    json_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
        )
        self._compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )
        if self.grammar_type == "json":
            ctx = self._compiler.compile_json_schema(self.grammar_spec)
        elif self.grammar_type == "json_object":
            ctx = self._compiler.compile_json_schema('{"type":"object"}')
        elif self.grammar_type == "regex":
            ctx = self._compiler.compile_regex(self.grammar_spec)
        elif self.grammar_type == "grammar":
            ctx = self._compiler.compile_grammar(self.grammar_spec)
        else:
            raise ValueError(
                "unsupported grammar-backed structured output type: "
                f"{self.grammar_type}"
            )
        self._matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=0)

    def apply(self, logits: torch.Tensor, request: RequestState) -> torch.Tensor:
        masked = logits.detach().clone().unsqueeze(0)
        bitmask = self._bitmask.to(device=masked.device)
        bitmask.fill_(-1)
        self._matcher.fill_next_token_bitmask(bitmask, 0)
        xgr.apply_token_bitmask_inplace(masked, bitmask)
        return masked.squeeze(0)

    def on_token(self, request: RequestState, token_id: int) -> bool:
        del request
        return bool(self._matcher.accept_token(int(token_id)))

    def should_finish(self, request: RequestState) -> bool:
        text = _decode_text(self.tokenizer, list(request.generated_ids))
        if self.finish_mode == "choice":
            return text in set(self.choices or [])
        if self.finish_mode == "json_object":
            return _parse_complete_json_object(text) is not None
        if self.finish_mode == "json":
            parsed = _parse_complete_json_object(text)
            if parsed is None:
                return False
            if self.json_schema is None:
                return True
            try:
                validate_json_schema(parsed, self.json_schema)
            except ValidationError:
                return False
            return True
        if self.finish_mode == "regex":
            return bool(re.fullmatch(self.grammar_spec, text))
        return bool(self._matcher.is_terminated())


def _decode_text(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
    except TypeError:
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(token_ids)


def _json_prefix_state(text: str) -> tuple[bool, bool]:
    stripped = text.lstrip()
    if not stripped:
        return True, False
    if not stripped.startswith("{"):
        return False, False

    in_string = False
    escape = False
    object_depth = 0
    array_depth = 0
    started = False
    complete = False
    prev_sig: str | None = None

    for idx, ch in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                prev_sig = '"'
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if prev_sig not in (None, ":", "[", ","):
                return False, False
            object_depth += 1
            started = True
            prev_sig = ch
            continue
        if ch == "}":
            if prev_sig in (":", ","):
                return False, False
            object_depth -= 1
            if object_depth < 0:
                return False, False
            if started and object_depth == 0 and array_depth == 0:
                rest = stripped[idx + 1 :]
                if rest.strip():
                    return False, False
                complete = True
            prev_sig = ch
            continue
        if ch == "[":
            if prev_sig not in (":", "[", ","):
                return False, False
            array_depth += 1
            prev_sig = ch
            continue
        if ch == "]":
            if prev_sig in (":", ","):
                return False, False
            array_depth -= 1
            if array_depth < 0:
                return False, False
            prev_sig = ch
            continue
        if ch == ":":
            if prev_sig != '"':
                return False, False
            prev_sig = ch
            continue
        if ch == ",":
            if prev_sig in (None, "{", "[", ":", ","):
                return False, False
            prev_sig = ch
            continue
        if not ch.isspace():
            prev_sig = ch

    if object_depth < 0 or array_depth < 0:
        return False, False
    if escape:
        return True, False
    return True, complete


def _parse_complete_json_object(text: str) -> dict[str, Any] | None:
    is_valid_prefix, is_complete = _json_prefix_state(text)
    if not is_valid_prefix or not is_complete:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


@dataclass
class ChoiceStructuredOutputConstraint(LiteStructuredOutputConstraint):
    tokenizer: Any
    choices: list[str]

    def __post_init__(self) -> None:
        self.choice_token_ids = [
            list(self.tokenizer.encode(choice)) for choice in self.choices
        ]
        if not self.choice_token_ids or any(not seq for seq in self.choice_token_ids):
            raise ValueError("structured_outputs.choice must contain non-empty choices")

    def apply(self, logits: torch.Tensor, request: RequestState) -> torch.Tensor:
        generated_ids = list(request.generated_ids)
        allowed = {
            seq[len(generated_ids)]
            for seq in self.choice_token_ids
            if len(seq) > len(generated_ids)
            and seq[: len(generated_ids)] == generated_ids
        }
        if not allowed:
            return logits
        masked = torch.full_like(logits, float("-inf"))
        index = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        masked[index] = logits[index]
        return masked

    def should_finish(self, request: RequestState) -> bool:
        generated_ids = list(request.generated_ids)
        return any(seq == generated_ids for seq in self.choice_token_ids)


@dataclass
class JsonStructuredOutputConstraint(LiteStructuredOutputConstraint):
    tokenizer: Any
    json_schema: dict[str, Any] | None = None
    candidate_limit: int = 256

    def apply(self, logits: torch.Tensor, request: RequestState) -> torch.Tensor:
        generated_ids = list(request.generated_ids)
        if logits.numel() == 0:
            return logits
        candidate_limit = min(int(self.candidate_limit), logits.numel())
        _, top_indices = torch.topk(logits, k=candidate_limit)
        allowed: list[int] = []
        for token_id in top_indices.tolist():
            text = _decode_text(self.tokenizer, generated_ids + [int(token_id)])
            is_valid_prefix, _ = _json_prefix_state(text)
            if is_valid_prefix:
                allowed.append(int(token_id))
        if not allowed:
            if not generated_ids:
                brace_tokens = self.tokenizer.encode("{")
                if brace_tokens:
                    allowed.append(int(brace_tokens[0]))
            if not allowed:
                return logits
        masked = torch.full_like(logits, float("-inf"))
        index = torch.tensor(
            sorted(set(allowed)), device=logits.device, dtype=torch.long
        )
        masked[index] = logits[index]
        return masked

    def should_finish(self, request: RequestState) -> bool:
        parsed = _parse_complete_json_object(
            _decode_text(self.tokenizer, list(request.generated_ids))
        )
        if parsed is None:
            return False
        if self.json_schema is None:
            return True
        try:
            validate_json_schema(parsed, self.json_schema)
        except ValidationError:
            return False
        return True


def build_structured_output_constraint(
    tokenizer: Any,
    sampling_params: SamplingParams,
) -> LiteStructuredOutputConstraint | None:
    params: StructuredOutputsParams | None = sampling_params.structured_outputs
    if params is None or params.all_constraints_none():
        return None
    try:
        return _build_xgrammar_constraint(tokenizer, params)
    except Exception:
        if params.regex is not None or params.grammar is not None:
            raise ValueError(
                "grammar-backed structured outputs require an "
                "xgrammar-compatible tokenizer/backend"
            ) from None
    if params.choice is not None:
        return ChoiceStructuredOutputConstraint(
            tokenizer=tokenizer, choices=list(params.choice)
        )
    if params.json_object:
        return JsonStructuredOutputConstraint(tokenizer=tokenizer)
    if params.json is not None:
        if isinstance(params.json, str):
            schema = json.loads(params.json)
        else:
            schema = params.json
        return JsonStructuredOutputConstraint(tokenizer=tokenizer, json_schema=schema)
    raise ValueError(
        "Lite structured outputs currently support structured_outputs.choice, "
        "structured_outputs.json_object, structured_outputs.json, "
        "and grammar-backed regex/grammar when xgrammar is available"
    )


def _build_xgrammar_constraint(
    tokenizer: Any,
    params: StructuredOutputsParams,
) -> LiteStructuredOutputConstraint:
    vocab_size = _estimate_vocab_size(tokenizer)
    if params.choice is not None:
        return XgrammarStructuredOutputConstraint(
            tokenizer=tokenizer,
            grammar_type="regex",
            grammar_spec=_choice_as_regex(list(params.choice)),
            vocab_size=vocab_size,
            finish_mode="choice",
            choices=list(params.choice),
        )
    if params.json_object:
        return XgrammarStructuredOutputConstraint(
            tokenizer=tokenizer,
            grammar_type="json_object",
            grammar_spec='{"type":"object"}',
            vocab_size=vocab_size,
            finish_mode="json_object",
        )
    if params.json is not None:
        schema = (
            params.json if isinstance(params.json, dict) else json.loads(params.json)
        )
        return XgrammarStructuredOutputConstraint(
            tokenizer=tokenizer,
            grammar_type="json",
            grammar_spec=json.dumps(schema),
            vocab_size=vocab_size,
            finish_mode="json",
            json_schema=schema,
        )
    if params.regex is not None:
        return XgrammarStructuredOutputConstraint(
            tokenizer=tokenizer,
            grammar_type="regex",
            grammar_spec=params.regex,
            vocab_size=vocab_size,
            finish_mode="regex",
        )
    if params.grammar is not None:
        return XgrammarStructuredOutputConstraint(
            tokenizer=tokenizer,
            grammar_type="grammar",
            grammar_spec=params.grammar,
            vocab_size=vocab_size,
        )
    raise ValueError("no structured output constraint to build")
