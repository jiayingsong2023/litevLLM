# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.sampling_driver import SamplingDriver
from vllm.engine.step_plan import AdmissionPlan, DecodePlan, PrefillPlan, StepPlan
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


def test_lite_single_gpu_backend_hook_defaults_are_noop() -> None:
    backend = LiteSingleGpuBackend(
        scheduler=object(),
        observer=object(),
        prefill_executor=object(),
        decode_executor=object(),
        sampling_driver=None,
        output_coordinator=None,
        kv_block_manager=KVBlockManager(
            kv_caches=[],
            kv_scale_caches=[],
            block_size=2,
            num_blocks_per_seq=1,
        ),
    )

    request_state = {"seq_len": 3}
    step_plan = object()
    scheduler = object()

    assert backend.maybe_apply_prefix_cache(request_state) is None
    assert backend.maybe_preempt(step_plan, scheduler) is step_plan
    assert request_state == {"seq_len": 3}
    assert backend.stats() == {
        "backend_type": "lite_single_gpu",
        "block_size": 2,
        "num_blocks_per_seq": 1,
        "num_kv_layers": 0,
        "min_prefix_cache_partial_hit_tokens": 1,
        "preemption_mode": "defer_prefill",
        "preemption_min_backlog": 1,
        "preemption_min_decodes": 1,
        "preemption_max_queue_wait_s": 0.0,
        "preemptible_service_classes": [],
        "preempt_multimodal_prefills": False,
        "preempt_multimodal_max_queue_wait_s": 0.0,
        "multimodal_prefix_cache_protect_threshold": 0.0,
        "prefix_cache_materialized_hits": 0,
        "prefix_cache_materialized_exact_hits": 0,
        "prefix_cache_materialized_partial_hits": 0,
        "prefix_cache_materialized_saved_prefill_tokens": 0,
        "prefix_cache_avg_saved_prefill_tokens_per_materialized_hit": 0.0,
        "prefix_cache": {
            "entries": 0,
            "capacity": 8,
            "lookups": 0,
            "exact_hits": 0,
            "partial_hits": 0,
            "misses": 0,
            "lookup_candidates_total": 0,
            "lookup_comparisons": 0,
            "hit_rate": 0.0,
            "avg_candidates_per_lookup": 0.0,
            "avg_comparisons_per_lookup": 0.0,
        },
        "multimodal": {},
    }


class _FakeObserver:
    def __init__(self) -> None:
        self.first_tokens = []
        self.finished = []
        self.prefix_cache_events = []
        self.preemption_events = []
        self.multimodal_preemption_guards = []

    def on_prefix_cache_event(
        self,
        request_id: str,
        *,
        hit: bool,
        exact: bool,
        prefix_len: int,
        saved_prefill_tokens: int,
        is_multimodal: bool = False,
    ) -> None:
        del is_multimodal
        self.prefix_cache_events.append(
            (request_id, hit, exact, prefix_len, saved_prefill_tokens)
        )

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
        multimodal_prefill_requests: int = 0,
    ) -> None:
        self.preemption_events.append(
            (preempted_prefill_requests, queued_backlog, multimodal_prefill_requests)
        )

    def on_multimodal_preemption_guard(
        self,
        *,
        protected_prefill_requests: int,
        prefix_cache_hit_rate: float,
    ) -> None:
        self.multimodal_preemption_guards.append(
            (protected_prefill_requests, prefix_cache_hit_rate)
        )

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        self.first_tokens.append((request_id, ttft_s))

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self.finished.append((request_id, reason))

    def on_prefill_executed(self, plan, num_outputs: int) -> None:
        del plan, num_outputs

    def on_decode_executed(self, plan, num_outputs: int) -> None:
        del plan, num_outputs


class _FakeScheduler:
    def __init__(self) -> None:
        self.requests = {}
        self.outputs = []
        self.freed = []

    def get_request(self, request_id: str):
        return self.requests[request_id]

    def publish_output(self, request_id: str, output) -> None:
        self.outputs.append((request_id, output))

    def free_request(self, request_id: str) -> None:
        self.freed.append(request_id)


class _FakeSamplingDriver:
    def sample_next_token(self, logits: torch.Tensor, request_state) -> int:
        del request_state
        return int(torch.argmax(logits).item())

    def completion_eos_ids(self, request_state) -> list[int]:
        del request_state
        return []


class _FakeOutputCoordinator:
    def finalize_step(self, request_id, request_state, next_token):
        return {
            "request_id": request_id,
            "token": next_token,
            "finished": request_state["finished"],
        }


def _backend(
    *,
    scheduler,
    observer,
    prefill_executor=object(),
    decode_executor=object(),
    sampling_driver=None,
    output_coordinator=None,
    kv_caches=None,
    kv_scale_caches=None,
    block_size: int = 2,
    num_blocks_per_seq: int = 1,
    **kwargs,
):
    kv_caches = [] if kv_caches is None else kv_caches
    kv_scale_caches = [] if kv_scale_caches is None else kv_scale_caches
    return LiteSingleGpuBackend(
        scheduler=scheduler,
        observer=observer,
        prefill_executor=prefill_executor,
        decode_executor=decode_executor,
        sampling_driver=sampling_driver,
        output_coordinator=output_coordinator,
        kv_block_manager=KVBlockManager(
            kv_caches=kv_caches,
            kv_scale_caches=kv_scale_caches,
            block_size=block_size,
            num_blocks_per_seq=num_blocks_per_seq,
        ),
        **kwargs,
    )


class _ChoiceTokenizer:
    eos_token_id = 99

    def encode(self, text: str):
        mapping = {
            "A": [1],
            "B": [2],
            "C": [3],
            "AB": [1, 2],
            "AC": [1, 3],
            "{": [4],
            "}": [5],
            "\"": [6],
            ":": [7],
            ",": [8],
            "1": [9],
            " ": [10],
            "{\"a\":1}": [4, 6, 1, 6, 7, 9, 5],
        }
        return list(mapping[text])

    def decode(self, token_ids, **kwargs):
        del kwargs
        inv = {1: "A", 2: "B", 3: "C", 4: "{", 5: "}", 6: "\"", 7: ":", 8: ",", 9: "1", 10: " ", 99: ""}
        return "".join(inv.get(int(t), "?") for t in token_ids)


class _ChoicePolicies:
    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids,
        sampling_params,
        capital_question_bias_token_ids,
        is_chinese_capital_question,
    ) -> torch.Tensor:
        del generated_ids, sampling_params, capital_question_bias_token_ids, is_chinese_capital_question
        return logits

    def should_early_stop(self, generated_ids, current_text: str) -> bool:
        del generated_ids, current_text
        return False

    def cleanup_output_text(self, text: str) -> str:
        return text


def _hf_tokenizer():
    vocab = {
        "[UNK]": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "AB": 4,
        "AC": 5,
        "{": 6,
        "}": 7,
        "\"": 8,
        ":": 9,
        ",": 10,
        "1": 11,
        "x": 12,
        "y": 13,
        '{"A":1}': 14,
    }
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        eos_token="[UNK]",
    )


def test_lite_single_gpu_backend_materializes_prefix_cache_hit() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=_FakeSamplingDriver(),
        output_coordinator=_FakeOutputCoordinator(),
        kv_caches=[(torch.zeros((4, 2, 1, 1)), torch.zeros((4, 2, 1, 1)))],
        kv_scale_caches=[(None, None)],
        block_size=2,
        num_blocks_per_seq=2,
        max_prefix_cache_entries=2,
    )

    source_req = {
        "request_id": "src",
        "input_ids": [11, 12, 13],
        "slot_idx": 0,
        "seq_len": 3,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 0.0,
        "first_token_at": None,
    }
    backend.kv_block_manager.kv_caches[0][0][0:2].copy_(
        torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    )
    backend.kv_block_manager.kv_caches[0][1][0:2].copy_(
        torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]])
    )
    backend._store_prefix_cache_entry("src", source_req, torch.tensor([0.1, 0.9]))

    new_req = {
        "request_id": "dst",
        "input_ids": [11, 12, 13],
        "slot_idx": None,
        "seq_len": 0,
        "is_prefill": True,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 1.0,
        "first_token_at": None,
    }
    scheduler.requests["dst"] = new_req

    assert backend.maybe_apply_prefix_cache(new_req) is None
    assert "_prefix_cache_entry" in new_req
    assert new_req["generated_ids"] == []

    new_req["slot_idx"] = 1
    backend.maybe_apply_prefix_cache(new_req)

    assert new_req["seq_len"] == 3
    assert new_req["is_prefill"] is False
    assert new_req["generated_ids"] == [1]
    assert torch.equal(
        backend.kv_block_manager.kv_caches[0][0][2:4],
        torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]),
    )
    assert torch.equal(
        backend.kv_block_manager.kv_caches[0][1][2:4],
        torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]]),
    )
    assert scheduler.outputs == [("dst", {"request_id": "dst", "token": 1, "finished": False})]
    assert observer.first_tokens and observer.first_tokens[0][0] == "dst"
    assert observer.prefix_cache_events == [("dst", True, True, 3, 3)]
    assert backend.stats()["prefix_cache_materialized_hits"] == 1
    assert backend.stats()["prefix_cache_materialized_exact_hits"] == 1
    assert backend.stats()["prefix_cache_materialized_partial_hits"] == 0
    assert backend.stats()["prefix_cache_materialized_saved_prefill_tokens"] == 3


def test_lite_single_gpu_backend_materializes_longest_prefix_without_token() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=_FakeSamplingDriver(),
        output_coordinator=_FakeOutputCoordinator(),
        kv_caches=[(torch.zeros((4, 2, 1, 1)), torch.zeros((4, 2, 1, 1)))],
        kv_scale_caches=[(None, None)],
        block_size=2,
        num_blocks_per_seq=2,
        max_prefix_cache_entries=2,
    )

    source_req = {
        "request_id": "src",
        "input_ids": [11, 12, 13, 14],
        "slot_idx": 0,
        "seq_len": 4,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 0.0,
        "first_token_at": None,
    }
    backend.kv_block_manager.kv_caches[0][0][0:2].copy_(
        torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    )
    backend.kv_block_manager.kv_caches[0][1][0:2].copy_(
        torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]])
    )
    backend._store_prefix_cache_entry("src", source_req, torch.tensor([0.1, 0.9]))

    new_req = {
        "request_id": "dst",
        "input_ids": [11, 12, 13, 99, 100],
        "slot_idx": None,
        "seq_len": 0,
        "is_prefill": True,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 1.0,
        "first_token_at": None,
    }
    scheduler.requests["dst"] = new_req

    backend.maybe_apply_prefix_cache(new_req)
    assert new_req["_prefix_cache_hit_len"] == 3
    assert "_prefix_cache_entry" in new_req
    assert new_req["generated_ids"] == []

    new_req["slot_idx"] = 1
    backend.maybe_apply_prefix_cache(new_req)

    assert new_req["seq_len"] == 3
    assert new_req["is_prefill"] is True
    assert new_req["generated_ids"] == []
    assert scheduler.outputs == []
    assert observer.first_tokens == []
    assert observer.prefix_cache_events == [("dst", True, False, 3, 3)]
    assert backend.stats()["prefix_cache_materialized_hits"] == 1
    assert backend.stats()["prefix_cache_materialized_exact_hits"] == 0
    assert backend.stats()["prefix_cache_materialized_partial_hits"] == 1
    assert backend.stats()["prefix_cache_materialized_saved_prefill_tokens"] == 3


def test_lite_single_gpu_backend_multimodal_prefix_cache_does_not_cross_image_namespace() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        kv_caches=[(torch.zeros((4, 2, 1, 1)), torch.zeros((4, 2, 1, 1)))],
        kv_scale_caches=[(None, None)],
        block_size=2,
        num_blocks_per_seq=2,
        max_prefix_cache_entries=2,
    )

    source_req = {
        "request_id": "src-mm",
        "input_ids": [11, 12, 13],
        "slot_idx": 0,
        "seq_len": 3,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 0.0,
        "first_token_at": None,
        "multi_modal_data": {"image": [{"image": "file:///tmp/cat-a.png"}]},
        "is_multimodal": True,
    }
    backend._store_prefix_cache_entry("src-mm", source_req, torch.tensor([0.1, 0.9]))

    other_image_req = {
        "request_id": "dst-mm",
        "input_ids": [11, 12, 13],
        "slot_idx": None,
        "seq_len": 0,
        "is_prefill": True,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 1.0,
        "first_token_at": None,
        "multi_modal_data": {"image": [{"image": "file:///tmp/cat-b.png"}]},
        "is_multimodal": True,
    }

    assert backend.maybe_apply_prefix_cache(other_image_req) is None
    assert "_prefix_cache_entry" not in other_image_req
    assert observer.prefix_cache_events == [("dst-mm", False, False, 0, 0)]


def test_lite_single_gpu_backend_multimodal_prefix_cache_hits_for_same_image_namespace() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=_FakeSamplingDriver(),
        output_coordinator=_FakeOutputCoordinator(),
        kv_caches=[(torch.zeros((4, 2, 1, 1)), torch.zeros((4, 2, 1, 1)))],
        kv_scale_caches=[(None, None)],
        block_size=2,
        num_blocks_per_seq=2,
        max_prefix_cache_entries=2,
    )

    source_req = {
        "request_id": "src-mm",
        "input_ids": [11, 12, 13],
        "slot_idx": 0,
        "seq_len": 3,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 0.0,
        "first_token_at": None,
        "multi_modal_data": {"image": [{"image": "file:///tmp/cat-a.png"}]},
        "is_multimodal": True,
    }
    backend.kv_block_manager.kv_caches[0][0][0:2].copy_(
        torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    )
    backend.kv_block_manager.kv_caches[0][1][0:2].copy_(
        torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]])
    )
    backend._store_prefix_cache_entry("src-mm", source_req, torch.tensor([0.1, 0.9]))

    same_image_req = {
        "request_id": "dst-mm",
        "input_ids": [11, 12, 13],
        "slot_idx": None,
        "seq_len": 0,
        "is_prefill": True,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 1.0,
        "first_token_at": None,
        "multi_modal_data": {"image": [{"image": "file:///tmp/cat-a.png"}]},
        "is_multimodal": True,
    }
    scheduler.requests["dst-mm"] = same_image_req

    assert backend.maybe_apply_prefix_cache(same_image_req) is None
    assert "_prefix_cache_entry" in same_image_req
    same_image_req["slot_idx"] = 1
    backend.maybe_apply_prefix_cache(same_image_req)

    assert same_image_req["seq_len"] == 3
    assert same_image_req["generated_ids"] == [1]
    assert observer.prefix_cache_events == [("dst-mm", True, True, 3, 3)]


def test_lite_single_gpu_backend_reports_prefix_cache_miss() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        max_prefix_cache_entries=2,
    )

    request_state = {
        "request_id": "miss",
        "input_ids": [101, 102],
        "slot_idx": None,
        "seq_len": 0,
    }

    assert backend.maybe_apply_prefix_cache(request_state) is None
    assert observer.prefix_cache_events == [("miss", False, False, 0, 0)]
    assert backend.stats()["prefix_cache"]["misses"] == 1


def test_lite_single_gpu_backend_ignores_too_short_partial_prefix_hit() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        max_prefix_cache_entries=2,
        min_prefix_cache_partial_hit_tokens=3,
    )

    backend.prefix_cache.put(
        type(
            "_Entry",
            (),
            {
                "key": (11, 12, 13),
                "prompt_len": 3,
                "used_blocks": 1,
                "k_blocks": [],
                "v_blocks": [],
                "k_scale_blocks": [],
                "v_scale_blocks": [],
                "last_prompt_logits": torch.tensor([1.0]),
            },
        )()
    )
    request_state = {
        "request_id": "short-partial",
        "input_ids": [11, 12, 99],
        "slot_idx": None,
        "seq_len": 0,
    }

    assert backend.maybe_apply_prefix_cache(request_state) is None
    assert "_prefix_cache_entry" not in request_state
    assert observer.prefix_cache_events == [("short-partial", False, False, 0, 0)]


def test_lite_single_gpu_backend_stats_include_prefix_cache_entries() -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        sampling_driver=_FakeSamplingDriver(),
        output_coordinator=_FakeOutputCoordinator(),
        kv_caches=[(torch.zeros((2, 2, 1, 1)), torch.zeros((2, 2, 1, 1)))],
        kv_scale_caches=[(None, None)],
        block_size=2,
        num_blocks_per_seq=1,
        max_prefix_cache_entries=3,
    )

    source_req = {
        "request_id": "src",
        "input_ids": [11, 12],
        "slot_idx": 0,
        "seq_len": 2,
        "generated_ids": [],
        "sampling_params": type("SP", (), {"max_tokens": 4})(),
        "finished": False,
        "admitted_at": 0.0,
        "first_token_at": None,
    }
    backend._store_prefix_cache_entry("src", source_req, torch.tensor([0.1, 0.9]))

    stats = backend.stats()

    assert stats["backend_type"] == "lite_single_gpu"
    assert stats["block_size"] == 2
    assert stats["num_blocks_per_seq"] == 1
    assert stats["num_kv_layers"] == 1
    assert stats["min_prefix_cache_partial_hit_tokens"] == 1
    assert stats["preemption_mode"] == "defer_prefill"
    assert stats["preemption_max_queue_wait_s"] == 0.0
    assert stats["prefix_cache_materialized_hits"] == 0
    assert stats["prefix_cache_materialized_exact_hits"] == 0
    assert stats["prefix_cache_materialized_partial_hits"] == 0
    assert stats["prefix_cache_materialized_saved_prefill_tokens"] == 0
    assert stats["prefix_cache_avg_saved_prefill_tokens_per_materialized_hit"] == 0.0
    assert stats["prefix_cache"] == {
        "entries": 1,
        "capacity": 3,
        "lookups": 0,
        "exact_hits": 0,
        "partial_hits": 0,
        "misses": 0,
        "lookup_candidates_total": 0,
        "lookup_comparisons": 0,
        "hit_rate": 0.0,
        "avg_candidates_per_lookup": 0.0,
        "avg_comparisons_per_lookup": 0.0,
    }


def test_lite_single_gpu_backend_preemption_respects_queue_wait_guardrail() -> None:
    observer = _FakeObserver()
    scheduler = _FakeScheduler()
    scheduler.queued_request_count = 2
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        preemption_max_queue_wait_s=2.0,
    )

    step_plan = StepPlan(
        admissions=AdmissionPlan(request_ids=[]),
        prefills=PrefillPlan(request_ids=["p0"], chunk_len=1, token_budget=1),
        decodes=DecodePlan(request_ids=["d0"], token_budget=1, use_fast_path=False),
        step_token_budget=2,
        queued_before=2,
        running_before=2,
        queued_p95_wait_s=3.0,
    )

    result = backend.maybe_preempt(step_plan, scheduler)

    assert result is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_preemption_respects_preemptible_service_classes() -> None:
    observer = _FakeObserver()
    scheduler = _FakeScheduler()
    scheduler.queued_request_count = 2
    scheduler.requests["p0"] = {"service_class": "latency"}
    backend = _backend(
        scheduler=scheduler,
        observer=observer,
        preemptible_service_classes={"background"},
    )

    step_plan = StepPlan(
        admissions=AdmissionPlan(request_ids=[]),
        prefills=PrefillPlan(request_ids=["p0"], chunk_len=1, token_budget=1),
        decodes=DecodePlan(request_ids=["d0"], token_budget=1, use_fast_path=False),
        step_token_budget=2,
        queued_before=2,
        running_before=2,
    )

    result = backend.maybe_preempt(step_plan, scheduler)

    assert result is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_reset_stats_can_clear_prefix_cache() -> None:
    backend = _backend(
        scheduler=object(),
        observer=object(),
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        max_prefix_cache_entries=2,
    )
    backend.prefix_cache.put(
        type(
            "_Entry",
            (),
            {
                "key": (1, 2),
                "prompt_len": 2,
                "used_blocks": 1,
                "k_blocks": [],
                "v_blocks": [],
                "k_scale_blocks": [],
                "v_scale_blocks": [],
                "last_prompt_logits": torch.tensor([1.0]),
            },
        )()
    )

    backend.reset_stats(clear_prefix_cache=True)

    assert backend.stats()["prefix_cache"] == {
        "entries": 0,
        "capacity": 2,
        "lookups": 0,
        "exact_hits": 0,
        "partial_hits": 0,
        "misses": 0,
        "lookup_candidates_total": 0,
        "lookup_comparisons": 0,
        "hit_rate": 0.0,
        "avg_candidates_per_lookup": 0.0,
        "avg_comparisons_per_lookup": 0.0,
    }


def test_lite_single_gpu_backend_soft_preempts_prefill_under_backlog() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
    )

    class _Sched:
        queued_request_count = 2

    step_plan = StepPlan(
        admissions=AdmissionPlan(request_ids=["q2"]),
        prefills=PrefillPlan(request_ids=["p1"], chunk_len=4, token_budget=4),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=2,
        running_before=2,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is not step_plan
    assert out.prefills is None
    assert out.decodes == step_plan.decodes
    assert out.admissions == step_plan.admissions
    assert observer.preemption_events == [(1, 2, 0)]


def test_lite_single_gpu_backend_does_not_preempt_without_backlog() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
    )

    class _Sched:
        queued_request_count = 0

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["p1"], chunk_len=4, token_budget=4),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=0,
        running_before=2,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_preemption_thresholds_gate_soft_preemption() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        preemption_min_backlog=2,
        preemption_min_decodes=2,
    )

    class _Sched:
        queued_request_count = 1

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["p1"], chunk_len=4, token_budget=4),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=1,
        running_before=2,
    )

    assert backend.maybe_preempt(step_plan, _Sched()) is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_respects_prefill_starvation_protection() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
    )

    class _Sched:
        queued_request_count = 3

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["p1"], chunk_len=2, token_budget=2),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=3,
        running_before=2,
        prefill_starvation_protected=True,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_protects_multimodal_prefill_by_default() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
    )

    class _Sched:
        queued_request_count = 3

        @staticmethod
        def get_request(request_id: str):
            if request_id == "mm1":
                return {
                    "is_multimodal": True,
                    "multi_modal_data": {"image": [{"image": "file:///tmp/cat.png"}]},
                }
            return {"is_multimodal": False}

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["mm1"], chunk_len=2, token_budget=2),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=3,
        running_before=2,
        queued_multimodal_p95_wait_s=0.0,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is step_plan
    assert observer.preemption_events == []


def test_lite_single_gpu_backend_can_preempt_multimodal_prefill_under_pressure() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        preempt_multimodal_prefills=True,
        preempt_multimodal_max_queue_wait_s=0.5,
    )

    class _Sched:
        queued_request_count = 3

        @staticmethod
        def get_request(request_id: str):
            if request_id == "mm1":
                return {
                    "is_multimodal": True,
                    "multi_modal_data": {"image": [{"image": "file:///tmp/cat.png"}]},
                }
            return {"is_multimodal": False}

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["mm1"], chunk_len=2, token_budget=2),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=3,
        running_before=2,
        prefill_multimodal_requests=1,
        queued_multimodal_requests=2,
        queued_multimodal_p95_wait_s=1.0,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is not step_plan
    assert out.prefills is None
    assert observer.preemption_events == [(1, 3, 1)]


def test_lite_single_gpu_backend_protects_multimodal_prefix_prefill_on_high_hit_rate() -> None:
    observer = _FakeObserver()
    backend = _backend(
        scheduler=object(),
        observer=observer,
        sampling_driver=None,
        output_coordinator=None,
        block_size=2,
        num_blocks_per_seq=1,
        preempt_multimodal_prefills=True,
        preempt_multimodal_max_queue_wait_s=0.5,
        multimodal_prefix_cache_protect_threshold=0.8,
    )

    class _Sched:
        queued_request_count = 3

        @staticmethod
        def get_request(request_id: str):
            if request_id == "mm1":
                return {
                    "is_multimodal": True,
                    "multi_modal_data": {"image": [{"image": "file:///tmp/cat.png"}]},
                }
            return {"is_multimodal": False}

    step_plan = StepPlan(
        admissions=None,
        prefills=PrefillPlan(request_ids=["mm1"], chunk_len=2, token_budget=2),
        decodes=DecodePlan(request_ids=["d1"], token_budget=1, use_fast_path=False),
        step_token_budget=8,
        queued_before=3,
        running_before=2,
        multimodal_prefix_cache_hit_rate=0.95,
        prefill_multimodal_requests=1,
        queued_multimodal_requests=2,
        queued_multimodal_p95_wait_s=1.0,
    )

    out = backend.maybe_preempt(step_plan, _Sched())

    assert out is step_plan
    assert observer.preemption_events == []
    assert observer.multimodal_preemption_guards == [(1, 0.95)]


def test_structured_output_choice_masks_sampling_and_finishes() -> None:
    tokenizer = _ChoiceTokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)

    request = {
        "request_id": "r1",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": SamplingParams(
            temperature=0.0,
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(choice=["AB", "AC"]),
        ),
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": __import__(
            "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
        ).build_structured_output_constraint(
            tokenizer,
            SamplingParams(
                temperature=0.0,
                max_tokens=4,
                structured_outputs=StructuredOutputsParams(choice=["AB", "AC"]),
            ),
        ),
    }

    token1 = sampling_driver.sample_next_token(torch.tensor([0.0, 5.0, 1.0, 0.5]), request)
    request["generated_ids"].append(token1)
    out1 = output.finalize_step("r1", request, token1)
    assert token1 == 1
    assert out1.finished is False

    token2 = sampling_driver.sample_next_token(torch.tensor([0.0, 0.1, 7.0, 3.0]), request)
    request["generated_ids"].append(token2)
    out2 = output.finalize_step("r1", request, token2)
    assert token2 == 2
    assert out2.finished is True


def test_structured_output_json_object_prefers_json_prefix_and_finishes() -> None:
    tokenizer = _ChoiceTokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rjson",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token_ids = [4, 6, 1, 6, 7, 9, 5]
    logits_seq = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 6.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ]

    for expected, logits in zip(token_ids, logits_seq):
        token = sampling_driver.sample_next_token(logits, request)
        request["generated_ids"].append(token)
        out = output.finalize_step("rjson", request, token)
        assert token == expected

    assert out.finished is True
    assert out.outputs[0].text == '{"A":1}'


def test_structured_output_json_schema_finishes_only_when_schema_valid() -> None:
    tokenizer = _ChoiceTokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(
            json='{"type":"object","properties":{"A":{"type":"string"}},"required":["A"]}'
        ),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rschema",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [4, 6, 1, 6, 7, 9, 5],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    out = output.finalize_step("rschema", request, 5)
    assert out.finished is False


def test_structured_output_regex_uses_xgrammar_backend() -> None:
    tokenizer = _hf_tokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(regex="^(?:AB|AC)$"),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rregex",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token1 = sampling_driver.sample_next_token(
        torch.tensor([0.0, 1.0, 2.0, 0.5, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        request,
    )
    request["generated_ids"].append(token1)
    out1 = output.finalize_step("rregex", request, token1)
    assert token1 in (4, 5)
    assert out1.finished is True


def test_structured_output_grammar_uses_xgrammar_backend() -> None:
    tokenizer = _hf_tokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(grammar='root ::= "x"'),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rgrammar",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token = sampling_driver.sample_next_token(
        torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 1.0]),
        request,
    )
    request["generated_ids"].append(token)
    out = output.finalize_step("rgrammar", request, token)

    assert token == 12
    assert out.finished is False

    token2 = sampling_driver.sample_next_token(
        torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        request,
    )
    request["generated_ids"].append(token2)
    out2 = output.finalize_step("rgrammar", request, token2)
    assert token2 == 0
    assert out2.finished is True


def test_structured_output_choice_uses_xgrammar_backend_with_hf_tokenizer() -> None:
    tokenizer = _hf_tokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(choice=["AB", "AC"]),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rchoicehf",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token = sampling_driver.sample_next_token(
        torch.tensor([0.0, 0.1, 0.2, 0.3, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        request,
    )
    request["generated_ids"].append(token)
    out = output.finalize_step("rchoicehf", request, token)

    assert token in (4, 5)
    assert out.finished is True


def test_structured_output_json_object_uses_xgrammar_backend_with_hf_tokenizer() -> None:
    tokenizer = _hf_tokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rjsonhf",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token = sampling_driver.sample_next_token(
        torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.0]),
        request,
    )
    request["generated_ids"].append(token)
    out = output.finalize_step("rjsonhf", request, token)

    assert token == 14
    assert out.finished is True


def test_structured_output_json_schema_uses_xgrammar_backend_with_hf_tokenizer() -> None:
    tokenizer = _hf_tokenizer()
    sampling_driver = SamplingDriver(tokenizer, None, _ChoicePolicies())
    output = OutputPipeline(tokenizer, _ChoicePolicies(), sampling_driver)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(
            json='{"type":"object","properties":{"A":{"type":"number"}},"required":["A"]}'
        ),
    )
    constraint = __import__(
        "vllm.engine.structured_output_lite", fromlist=["build_structured_output_constraint"]
    ).build_structured_output_constraint(tokenizer, sp)

    request = {
        "request_id": "rschemahf",
        "prompt": "p",
        "input_ids": [10],
        "generated_ids": [],
        "sampling_params": sp,
        "finished": False,
        "low_info_hits": 0,
        "structured_output_constraint": constraint,
    }

    token = sampling_driver.sample_next_token(
        torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.0]),
        request,
    )
    request["generated_ids"].append(token)
    out = output.finalize_step("rschemahf", request, token)

    assert token == 14
    assert out.finished is True
