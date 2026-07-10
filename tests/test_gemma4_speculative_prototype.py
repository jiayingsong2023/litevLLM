# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_speculative_prototype.py"
    spec = importlib.util.spec_from_file_location("gemma4_speculative_prototype", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proto_mod() -> Any:
    return _load_module()


def _make_fixture(prompts: list[dict[str, Any]]) -> str:
    return json.dumps({"version": 1, "prompts": prompts})


def test_propose_ngram_finds_repeated_suffix(proto_mod: Any) -> None:
    prefix = [1, 2, 3, 4, 5]
    generated = [2, 3]
    # The suffix [2, 3] appeared earlier starting at index 1 in
    # [1,2,3,4,5,2,3]. The 3 tokens after that earlier occurrence are [4, 5, 2].
    result = proto_mod.propose_ngram(prefix, generated, k=3, ngram_min=2, ngram_max=4)
    assert result == [4, 5, 2]


def test_propose_ngram_prefers_longer_needle(proto_mod: Any) -> None:
    prefix = [10, 20, 30, 40]
    generated = [10, 20, 30]
    # n=3 needle [10,20,30] matches at index 0; return [40, 10].
    result = proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=3)
    assert result == [40, 10]


def test_propose_ngram_returns_empty_when_no_match(proto_mod: Any) -> None:
    result = proto_mod.propose_ngram([1, 2, 3], [4, 5], k=5, ngram_min=2, ngram_max=4)
    assert result == []


def test_propose_ngram_does_not_match_final_suffix(proto_mod: Any) -> None:
    prefix = [1, 2]
    generated = [3, 4]
    # The only [3,4] is the final suffix itself; no earlier occurrence exists.
    result = proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=2)
    assert result == []


def test_run_target_logits_tie_word_embeddings(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 5
    fake_weight = torch.randn(vocab_size, hidden_size)
    fake_hidden = torch.randn(1, seq_len, hidden_size)

    inner = type(
        "FakeInner",
        (),
        {
            "config": SimpleNamespace(
                tie_word_embeddings=True,
                final_logit_softcapping=30.0,
            ),
            "embed_tokens": SimpleNamespace(weight=fake_weight),
            "layers": [None, None],
            "__call__": lambda self, *args, **kwargs: fake_hidden,
            "parameters": lambda self: iter([fake_weight]),
        },
    )()
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3, 4, 5]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert logits.abs().max().item() <= 30.5


def test_run_target_logits_untied_lm_head(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 3
    fake_hidden = torch.randn(1, seq_len, hidden_size)
    lm_logits = torch.randn(1, seq_len, vocab_size)

    lm_head = type(
        "FakeLMHead",
        (),
        {"__call__": lambda self, hidden, lora_mapping: lm_logits},
    )()
    inner = type(
        "FakeInner",
        (),
        {
            "config": SimpleNamespace(
                tie_word_embeddings=False,
                final_logit_softcapping=None,
            ),
            "embed_tokens": SimpleNamespace(
                weight=torch.randn(vocab_size, hidden_size)
            ),
            "layers": [None],
            "__call__": lambda self, *args, **kwargs: fake_hidden,
            "parameters": lambda self: iter([torch.randn(1)]),
        },
    )()
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner, lm_head=lm_head),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert torch.equal(logits, lm_logits)


def _make_mock_llm(reference_generated: list[int]) -> Any:
    """Return a mock LLM whose generate() emits reference_generated."""
    completion = SimpleNamespace(token_ids=list(reference_generated), text="")
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[completion],
    )

    def _generate(prompts: list[str], sampling_params: Any) -> list[Any]:
        return [output]

    return SimpleNamespace(generate=_generate)


def _mock_run_target_logits(reference: list[int], prompt_len: int):
    """Return logits whose argmax matches the reference at each generated position.

    Sets the target logit at index p-1 for every input position p in
    [prompt_len, L], using reference[p-prompt_len] when within range, so the
    bonus token past the last proposed draft is also defined.
    """

    def _run(llm: Any, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids[0].tolist()
        vocab_size = max(reference + ids) + 10
        logits = torch.full((1, len(ids), vocab_size), -1e9)
        for p in range(prompt_len, len(ids) + 1):
            ref_idx = p - prompt_len
            if ref_idx < len(reference):
                logits[0, p - 1, reference[ref_idx]] = 1e6
        return logits

    return _run


def test_speculative_decode_bit_exact_when_drafts_match(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = [10, 11, 12, 13]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        # Perfect oracle: return the next reference tokens.
        start = len(generated)
        return reference[start : start + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["acceptance_rate"] == 1.0
    assert result["accepted_total"] <= len(reference)


def test_speculative_decode_recovers_on_mismatch(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = [10, 11, 12]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )

    calls: list[int] = []

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        calls.append(len(generated))
        if len(generated) == 0:
            return [99, reference[1]]  # first draft token mismatches
        return reference[len(generated) : len(generated) + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["accepted_total"] < result["proposed_total"]


def test_speculative_decode_stops_at_eos(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = [10, 999, 11, 12]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )
    monkeypatch.setattr(proto_mod, "_get_eos_token_ids", lambda llm: {999})

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        start = len(generated)
        return reference[start : start + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == [10, 999]
    assert 999 in result["token_ids"]
    assert result["accepted_total"] == 2
    assert result["proposed_total"] == 2


def test_speculative_decode_truncates_to_max_new_tokens(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that accepts all drafts plus a bonus must not exceed the budget."""
    reference = [10, 11, 12, 13]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        start = len(generated)
        return reference[start : start + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=3,
        num_draft_tokens=3,
    )

    assert len(result["token_ids"]) == 3
    assert result["token_ids"] == [10, 11, 12]


def test_baseline_greedy_splits_prefill_decode_timings(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """baseline_greedy should report separate prefill and decode timings."""

    class FakeOutput:
        def __init__(self, request_id: str, token_ids: list[int]) -> None:
            self.request_id = request_id
            self.finished = True
            self.outputs = [SimpleNamespace(token_ids=list(token_ids))]
            self.prompt_token_ids = [1, 2, 3]

    class FakeEngine:
        def __init__(self) -> None:
            self._active = 1
            self._request_id: str = ""

        @property
        def active_request_count(self) -> int:
            return self._active

        def add_request(self, request_id: str, *args: Any, **kwargs: Any) -> None:
            self._request_id = request_id

        def step(self) -> list[Any]:
            time.sleep(0.005)
            self._active = 0
            return [FakeOutput(self._request_id, [10, 11, 12])]

        def abort_request(self, *args: Any, **kwargs: Any) -> None:
            pass

    fake_engine = FakeEngine()
    fake_tokenizer = SimpleNamespace(
        encode=lambda text: [1, 2, 3],
        chat_template=None,
    )
    fake_llm = SimpleNamespace(
        engine=fake_engine,
        tokenizer=fake_tokenizer,
    )

    result = proto_mod.baseline_greedy(fake_llm, "hello", max_new_tokens=3)

    assert result["prompt_token_ids"] == [1, 2, 3]
    assert result["token_ids"] == [10, 11, 12]
    assert result["prefill_time_s"] > 0
    assert result["decode_time_s"] >= 0
    # Three generated tokens; decode TPS excludes the first prefill-produced token.
    assert result["decode_tps"] == pytest.approx(2.0 / result["decode_time_s"], rel=0.1)


def test_add_request_with_token_ids_builds_and_enqueues(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_add_request_with_token_ids initializes the builder and enqueues the request."""
    built_request = SimpleNamespace(input_ids=[], guarded_prompt="wrong")
    requests: list[Any] = []

    def fake_build(*, request_id: str, prompt: str, sampling_params: Any) -> Any:
        return built_request

    class FakeScheduler:
        def __init__(self) -> None:
            self.admitted: list[str] = []

        def enqueue_request(self, request_id: str, req: Any) -> None:
            requests.append((request_id, req))

        def admit_queued_requests(self, max_new: int) -> list[str]:
            self.admitted = ["draft_1"]
            return self.admitted

    fake_scheduler = FakeScheduler()
    fake_engine = SimpleNamespace(
        request_builder=SimpleNamespace(build=fake_build),
        scheduler=fake_scheduler,
    )
    fake_llm = SimpleNamespace(engine=fake_engine)

    proto_mod._add_request_with_token_ids(
        fake_llm,
        "draft_1",
        [1, 2, 3],
        SimpleNamespace(),
    )

    assert len(requests) == 1
    assert requests[0][0] == "draft_1"
    assert requests[0][1].input_ids == [1, 2, 3]
    assert requests[0][1].guarded_prompt == ""


def test_add_request_with_token_ids_initializes_lazy_builder(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When request_builder is None, the helper initializes it with a dummy request."""
    built_request = SimpleNamespace(input_ids=[], guarded_prompt="")

    def fake_build(*, request_id: str, prompt: str, sampling_params: Any) -> Any:
        return built_request

    class FakeScheduler:
        def enqueue_request(self, request_id: str, req: Any) -> None:
            pass

        def admit_queued_requests(self, max_new: int) -> list[str]:
            return ["draft_2"]

    added_requests: list[tuple[str, str, Any]] = []
    aborted_requests: list[str] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.request_builder: Any | None = None

        def add_request(self, request_id: str, prompt: str, sp: Any) -> None:
            added_requests.append((request_id, prompt, sp))
            self.request_builder = SimpleNamespace(build=fake_build)

        def abort_request(self, request_id: str) -> None:
            aborted_requests.append(request_id)

    fake_engine = FakeEngine()
    fake_engine.scheduler = FakeScheduler()
    fake_llm = SimpleNamespace(engine=fake_engine)

    proto_mod._add_request_with_token_ids(
        fake_llm,
        "draft_2",
        [4, 5, 6],
        SimpleNamespace(),
    )

    assert len(added_requests) == 1
    assert added_requests[0][0] == "__init_request_builder__"
    assert aborted_requests == ["__init_request_builder__"]
    assert built_request.input_ids == [4, 5, 6]


def test_generate_draft_tokens_from_ids_returns_expected_tokens(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_draft_tokens_from_ids drives the engine until k tokens are produced."""

    class FakeOutput:
        def __init__(self, request_id: str, step_count: int) -> None:
            self.request_id = request_id
            self.outputs = [SimpleNamespace(token_ids=[10, 11][:step_count])]
            self.finished = step_count >= 2
            self.prompt_token_ids = [1, 2, 3]

    class FakeScheduler:
        def __init__(self) -> None:
            self.last_request_id: str | None = None

        def enqueue_request(self, request_id: str, req: Any) -> None:
            self.last_request_id = request_id

        def admit_queued_requests(self, max_new: int) -> list[str]:
            assert self.last_request_id is not None
            return [self.last_request_id]

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = 0
            self.scheduler = FakeScheduler()
            self.request_builder = SimpleNamespace(
                build=lambda **kwargs: SimpleNamespace(input_ids=[], guarded_prompt="")
            )

        @property
        def active_request_count(self) -> int:
            return 1 if self.calls < 2 else 0

        def step(self) -> list[Any]:
            self.calls += 1
            return [FakeOutput(self.scheduler.last_request_id or "", self.calls)]

        def abort_request(self, *args: Any, **kwargs: Any) -> None:
            pass

    fake_llm = SimpleNamespace(engine=FakeEngine())

    result = proto_mod.generate_draft_tokens_from_ids(fake_llm, [1, 2, 3], k=2)

    assert result == [10, 11]


def test_cli_help_exits_zero(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["gemma4_speculative_prototype.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        proto_mod.main()
    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--target-model" in captured.out
    assert "--num-draft-tokens" in captured.out


def _patch_tokenizer_gate(proto_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1, 2, 3],
        decode=lambda ids: "hello",
        get_vocab=lambda: {"a": 0, "b": 1},
        added_tokens_encoder={},
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        chat_template=None,
    )
    monkeypatch.setattr(proto_mod, "_load_tokenizer", lambda model_path: fake_tokenizer)
    monkeypatch.setattr(
        proto_mod,
        "build_report",
        lambda *args, **kwargs: {"passed": True, "details": {}},
    )


def test_cli_writes_json(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [
                {
                    "id": "p1",
                    "text": "hello",
                    "context_len": 3,
                    "max_new_tokens": 2,
                }
            ]
        )
    )
    json_out = tmp_path / "out.json"

    reference = [10, 11]
    prompt = [1, 2, 3]

    def _mock_baseline(*args, **kwargs):
        return {
            "prompt_token_ids": list(prompt),
            "token_ids": list(reference),
            "prefill_time_s": 0.1,
            "decode_time_s": 0.05,
            "decode_tps": 20.0,
        }

    def _mock_speculate(*args, **kwargs):
        return {
            "token_ids": list(reference),
            "baseline_token_ids": [],
            "bit_exact": False,
            "accepted_total": 2,
            "proposed_total": 2,
            "acceptance_rate": 1.0,
            "target_forwards": 1,
            "verify_time_s": 0.0,
            "baseline_tps": 0.0,
            "speculative_tps": 0.0,
            "projected_tps": 0.0,
        }

    _patch_tokenizer_gate(proto_mod, monkeypatch)
    monkeypatch.setattr(proto_mod, "baseline_greedy", _mock_baseline)
    monkeypatch.setattr(proto_mod, "speculative_decode", _mock_speculate)
    monkeypatch.setattr(
        proto_mod,
        "LLM",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(model_dir),
            "--prompt-file",
            str(fixture_path),
            "--json-out",
            str(json_out),
        ],
    )

    rc = proto_mod.main()
    assert rc == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert data["summary"]["bit_exact_all"] is True
    assert "acceptance_rate" in data["prompts"][0]
    assert "projected_tps" in data["prompts"][0]
    assert data["summary"]["mean_baseline_decode_tps"] == 20.0


def test_cli_returns_error_on_mismatch(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [
                {
                    "id": "p1",
                    "text": "hello",
                    "context_len": 3,
                    "max_new_tokens": 2,
                }
            ]
        )
    )

    def _mock_baseline(*args, **kwargs):
        return {
            "prompt_token_ids": [1, 2, 3],
            "token_ids": [10, 11],
            "prefill_time_s": 0.1,
            "decode_time_s": 0.05,
            "decode_tps": 20.0,
        }

    def _mock_speculate(*args, **kwargs):
        return {
            "token_ids": [99, 100],
            "baseline_token_ids": [],
            "bit_exact": False,
            "accepted_total": 0,
            "proposed_total": 2,
            "acceptance_rate": 0.0,
            "target_forwards": 1,
            "verify_time_s": 0.0,
            "baseline_tps": 0.0,
            "speculative_tps": 0.0,
            "projected_tps": 0.0,
        }

    _patch_tokenizer_gate(proto_mod, monkeypatch)
    monkeypatch.setattr(proto_mod, "baseline_greedy", _mock_baseline)
    monkeypatch.setattr(proto_mod, "speculative_decode", _mock_speculate)
    monkeypatch.setattr(
        proto_mod,
        "LLM",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(model_dir),
            "--prompt-file",
            str(fixture_path),
        ],
    )

    rc = proto_mod.main()
    assert rc == 1


def test_cli_allows_mismatch_when_fail_flag_disabled(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [
                {
                    "id": "p1",
                    "text": "hello",
                    "context_len": 3,
                    "max_new_tokens": 2,
                }
            ]
        )
    )

    def _mock_baseline(*args, **kwargs):
        return {
            "prompt_token_ids": [1, 2, 3],
            "token_ids": [10, 11],
            "prefill_time_s": 0.1,
            "decode_time_s": 0.05,
            "decode_tps": 20.0,
        }

    def _mock_speculate(*args, **kwargs):
        return {
            "token_ids": [99, 100],
            "baseline_token_ids": [],
            "bit_exact": False,
            "accepted_total": 0,
            "proposed_total": 2,
            "acceptance_rate": 0.0,
            "target_forwards": 1,
            "verify_time_s": 0.0,
            "baseline_tps": 0.0,
            "speculative_tps": 0.0,
            "projected_tps": 0.0,
        }

    _patch_tokenizer_gate(proto_mod, monkeypatch)
    monkeypatch.setattr(proto_mod, "baseline_greedy", _mock_baseline)
    monkeypatch.setattr(proto_mod, "speculative_decode", _mock_speculate)
    monkeypatch.setattr(
        proto_mod,
        "LLM",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(model_dir),
            "--prompt-file",
            str(fixture_path),
            "--fail-on-mismatch",
            "false",
        ],
    )

    rc = proto_mod.main()
    assert rc == 0


def test_cli_with_draft_model_computes_effective_tps(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_dir = tmp_path / "fake-target"
    draft_dir = tmp_path / "fake-draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [
                {
                    "id": "p1",
                    "text": "hello",
                    "context_len": 3,
                    "max_new_tokens": 2,
                }
            ]
        )
    )

    def _mock_baseline(*args, **kwargs):
        return {
            "prompt_token_ids": [1, 2, 3],
            "token_ids": [10, 11],
            "prefill_time_s": 0.1,
            "decode_time_s": 0.05,
            "decode_tps": 10.0,
        }

    def _mock_speculate(*args, **kwargs):
        return {
            "token_ids": [10, 11],
            "baseline_token_ids": [],
            "bit_exact": True,
            "accepted_total": 1,
            "proposed_total": 2,
            "acceptance_rate": 0.5,
            "target_forwards": 1,
            "verify_time_s": 0.1,
            "baseline_tps": 10.0,
            "speculative_tps": 5.0,
            "projected_tps": 15.0,
        }

    _patch_tokenizer_gate(proto_mod, monkeypatch)
    monkeypatch.setattr(proto_mod, "baseline_greedy", _mock_baseline)
    monkeypatch.setattr(proto_mod, "speculative_decode", _mock_speculate)
    monkeypatch.setattr(
        proto_mod,
        "generate_draft_tokens_from_ids",
        lambda *args, **kwargs: [20, 21],
    )
    json_out = tmp_path / "out.json"
    monkeypatch.setattr(
        proto_mod,
        "LLM",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(target_dir),
            "--draft-model",
            str(draft_dir),
            "--prompt-file",
            str(fixture_path),
            "--json-out",
            str(json_out),
        ],
    )

    rc = proto_mod.main()
    assert rc == 0
    data = json.loads(json_out.read_text())
    assert data["summary"]["effective_tps"] == pytest.approx(2.0 / 0.1)


def test_cli_tokenizer_gate_failure_returns_two(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_dir = tmp_path / "fake-target"
    target_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [{"id": "p1", "text": "hello", "context_len": 3, "max_new_tokens": 2}]
        )
    )

    fake_tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1, 2, 3],
        decode=lambda ids: "hello",
        get_vocab=lambda: {"a": 0},
        added_tokens_encoder={},
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        chat_template=None,
    )
    monkeypatch.setattr(proto_mod, "_load_tokenizer", lambda model_path: fake_tokenizer)
    monkeypatch.setattr(
        proto_mod,
        "build_report",
        lambda *args, **kwargs: {
            "passed": False,
            "details": {"reason": "vocab mismatch"},
        },
    )
    monkeypatch.setattr(
        proto_mod,
        "LLM",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(target_dir),
            "--prompt-file",
            str(fixture_path),
        ],
    )

    rc = proto_mod.main()
    assert rc == 2


def test_cli_memory_gate_failure_on_oom(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_dir = tmp_path / "fake-target"
    draft_dir = tmp_path / "fake-draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        _make_fixture(
            [{"id": "p1", "text": "hello", "context_len": 3, "max_new_tokens": 2}]
        )
    )

    calls: list[str] = []

    def fake_llm(**kwargs):
        calls.append(kwargs.get("model"))
        return SimpleNamespace(shutdown=lambda: None)

    _patch_tokenizer_gate(proto_mod, monkeypatch)
    monkeypatch.setattr(proto_mod, "LLM", fake_llm)

    class OOMError(Exception):
        pass

    monkeypatch.setattr(torch.cuda, "OutOfMemoryError", OOMError)

    def raise_oom(**kwargs):
        if kwargs.get("model") == str(draft_dir):
            raise OOMError("fake oom")
        return SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(proto_mod, "LLM", raise_oom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(target_dir),
            "--draft-model",
            str(draft_dir),
            "--prompt-file",
            str(fixture_path),
        ],
    )

    rc = proto_mod.main()
    assert rc == 2
