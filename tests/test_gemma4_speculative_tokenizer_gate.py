# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Gemma4 P1 speculative tokenizer gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_tokenizer_gate_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_speculative_tokenizer_gate.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_speculative_tokenizer_gate", p
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate_mod() -> Any:
    return _load_tokenizer_gate_module()


class FakeTokenizer:
    """Minimal tokenizer double for offline gate tests."""

    def __init__(
        self,
        vocab_size: int = 256000,
        bos_token_id: int | list[int] | None = 2,
        eos_token_id: int | list[int] | None = 1,
        pad_token_id: int | list[int] | None = 0,
        encode_map: dict[str, list[int]] | None = None,
        chat_template: str
        | None = "{% for m in messages %}{{ m['role'] }}{% endfor %}",
        chat_output: str = "<chat output>",
    ) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.chat_template = chat_template
        self._encode_map = encode_map or {}
        self._chat_output = chat_output

    def encode(
        self, text: str, *, add_special_tokens: bool = False, **_: Any
    ) -> list[int]:
        return list(self._encode_map.get(text, [ord(c) for c in text]))

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **_: Any,
    ) -> str:
        return self._chat_output


def _identical_tokenizers() -> tuple[FakeTokenizer, FakeTokenizer]:
    target = FakeTokenizer()
    draft = FakeTokenizer()
    return target, draft


def test_all_checks_pass_when_tokenizers_identical(gate_mod: Any) -> None:
    target, draft = _identical_tokenizers()
    report = gate_mod.build_report("target", "draft", target, draft, ["hello", "world"])

    assert report["vocab_size_match"] is True
    assert report["special_tokens_match"] is True
    assert report["encode_match"] is True
    assert report["chat_template_match"] is True
    assert report["passed"] is True


def test_vocab_size_mismatch_fails(gate_mod: Any) -> None:
    target, draft = _identical_tokenizers()
    draft.vocab_size = target.vocab_size + 1
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["vocab_size_match"] is False
    assert report["passed"] is False


def test_special_token_mismatch_fails(gate_mod: Any) -> None:
    target, draft = _identical_tokenizers()
    draft.eos_token_id = 999
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["special_tokens_match"] is False
    assert report["passed"] is False


def test_special_token_list_normalization(gate_mod: Any) -> None:
    target = FakeTokenizer(eos_token_id=[1, 2])
    draft = FakeTokenizer(eos_token_id=[2, 1])
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["special_tokens_match"] is True


def test_encode_mismatch_fails(gate_mod: Any) -> None:
    target = FakeTokenizer(encode_map={"hello": [10, 20, 30]})
    draft = FakeTokenizer(encode_map={"hello": [10, 21, 30]})
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["encode_match"] is False
    assert report["passed"] is False


def test_chat_template_mismatch_fails(gate_mod: Any) -> None:
    target = FakeTokenizer(chat_output="target_chat")
    draft = FakeTokenizer(chat_output="draft_chat")
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["chat_template_match"] is False
    assert report["passed"] is False


def test_chat_template_absent_returns_none(gate_mod: Any) -> None:
    target = FakeTokenizer(chat_template=None)
    draft = FakeTokenizer(chat_template=None)
    report = gate_mod.build_report("target", "draft", target, draft, ["hello"])

    assert report["chat_template_match"] is None
    assert report["passed"] is True


def test_cli_exit_code_zero_on_pass(
    gate_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, draft = _identical_tokenizers()
    monkeypatch.setattr(
        gate_mod, "_load_tokenizer", lambda path: target if "target" in path else draft
    )
    monkeypatch.setattr(
        sys, "argv", ["gate.py", "--target-model", "target", "--draft-model", "draft"]
    )

    rc = gate_mod.main()

    assert rc == 0


def test_cli_exit_code_one_on_fail(
    gate_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, draft = _identical_tokenizers()
    draft.vocab_size = target.vocab_size + 1
    monkeypatch.setattr(
        gate_mod, "_load_tokenizer", lambda path: target if "target" in path else draft
    )
    monkeypatch.setattr(
        sys, "argv", ["gate.py", "--target-model", "target", "--draft-model", "draft"]
    )

    rc = gate_mod.main()

    assert rc == 1


def test_cli_writes_json_report(
    gate_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target, draft = _identical_tokenizers()
    monkeypatch.setattr(
        gate_mod, "_load_tokenizer", lambda path: target if "target" in path else draft
    )
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate.py",
            "--target-model",
            "target",
            "--draft-model",
            "draft",
            "--json-out",
            str(out),
        ],
    )

    rc = gate_mod.main()

    assert rc == 0
    data = json.loads(out.read_text())
    assert data["target_model"] == "target"
    assert data["draft_model"] == "draft"
    assert data["passed"] is True
    assert "details" in data
