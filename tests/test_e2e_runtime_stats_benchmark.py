# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tests.e2e_full_benchmark import (
    _collect_runtime_stats,
    _derive_runtime_phase_diffs,
    _format_runtime_phase_diff_summary,
)


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def stats(self):
        self.calls += 1
        return {
            "scheduler": {"active_request_count": 1},
            "observer": {
                "admitted_requests": 3,
                "step_count": self.calls,
                "prefix_cache": {
                    "events": 2,
                    "hits": 1,
                    "misses": 1,
                    "exact_hits": 1,
                    "partial_hits": 0,
                    "saved_prefill_tokens": 6,
                },
                "preemption": {
                    "events": 1,
                    "preempted_steps": 1,
                    "preempted_prefill_requests": 2,
                },
                "fairness": {
                    "starvation_protected_steps": 1,
                    "fairness_guardrail_triggered_steps": 1,
                    "avg_admitted_queue_wait_s": 0.25,
                    "p95_queue_wait_s": 0.5,
                    "max_queue_wait_s": 1.0,
                    "per_class_p95_queue_wait_s": {
                        "latency": 0.5,
                        "background": 1.0,
                    },
                },
            },
            "backend": {
                "prefix_cache_materialized_hits": 1,
                "prefix_cache_materialized_saved_prefill_tokens": 4,
                "prefix_cache": {
                    "lookup_comparisons": 10,
                    "lookup_candidates_total": 3,
                    "hit_rate": 0.5,
                },
            },
        }


def test_collect_runtime_stats_copies_snapshot() -> None:
    llm = _FakeLLM()

    snap = _collect_runtime_stats(llm, phase="benchmark")

    assert snap["phase"] == "benchmark"
    assert snap["stats"] == {
        "scheduler": {"active_request_count": 1},
        "observer": {
            "admitted_requests": 3,
            "step_count": 1,
            "prefix_cache": {
                "events": 2,
                "hits": 1,
                "misses": 1,
                "exact_hits": 1,
                "partial_hits": 0,
                "saved_prefill_tokens": 6,
            },
            "preemption": {
                "events": 1,
                "preempted_steps": 1,
                "preempted_prefill_requests": 2,
            },
            "fairness": {
                "starvation_protected_steps": 1,
                "fairness_guardrail_triggered_steps": 1,
                "avg_admitted_queue_wait_s": 0.25,
                "p95_queue_wait_s": 0.5,
                "max_queue_wait_s": 1.0,
                "per_class_p95_queue_wait_s": {
                    "latency": 0.5,
                    "background": 1.0,
                },
            },
        },
        "backend": {
            "prefix_cache_materialized_hits": 1,
            "prefix_cache_materialized_saved_prefill_tokens": 4,
            "prefix_cache": {
                "lookup_comparisons": 10,
                "lookup_candidates_total": 3,
                "hit_rate": 0.5,
            },
        },
    }
    assert snap["derived_metrics"] == {
        "prefix_cache": {
            "request_count": 2.0,
            "lookup_hit_rate": 0.5,
            "materialized_hit_rate": 0.5,
            "saved_prefill_tokens_per_request": 2.0,
            "saved_prefill_tokens_per_materialized_hit": 4.0,
            "lookup_cost_per_request": 5.0,
            "lookup_candidates_per_request": 1.5,
        },
        "preemption": {
            "step_count": 1.0,
            "preempted_steps": 1.0,
            "preempted_prefill_requests": 2.0,
            "preempted_step_rate": 1.0,
            "preempted_prefill_requests_per_step": 2.0,
        },
        "fairness": {
            "step_count": 1.0,
            "admitted_requests": 3.0,
            "starvation_protected_steps": 1.0,
            "fairness_guardrail_triggered_steps": 1.0,
            "starvation_protected_step_rate": 1.0,
            "fairness_guardrail_triggered_step_rate": 1.0,
            "avg_admitted_queue_wait_s": 0.25,
            "p95_queue_wait_s": 0.5,
            "max_queue_wait_s": 1.0,
            "per_class_p95_queue_wait_s": {
                "latency": 0.5,
                "background": 1.0,
            },
        },
    }


def test_derive_runtime_phase_diffs_summarizes_benchmark_delta() -> None:
    diffs = _derive_runtime_phase_diffs(
        {
            "warmup": {
                "derived_metrics": {
                    "prefix_cache": {
                        "request_count": 1.0,
                        "lookup_hit_rate": 0.0,
                        "materialized_hit_rate": 0.0,
                        "saved_prefill_tokens_per_request": 0.0,
                        "saved_prefill_tokens_per_materialized_hit": 0.0,
                        "lookup_cost_per_request": 2.0,
                        "lookup_candidates_per_request": 1.0,
                    },
                    "preemption": {
                        "step_count": 2.0,
                        "preempted_steps": 0.0,
                        "preempted_prefill_requests": 0.0,
                        "preempted_step_rate": 0.0,
                        "preempted_prefill_requests_per_step": 0.0,
                    },
                    "fairness": {
                        "step_count": 2.0,
                        "admitted_requests": 2.0,
                        "starvation_protected_steps": 0.0,
                        "fairness_guardrail_triggered_steps": 0.0,
                        "starvation_protected_step_rate": 0.0,
                        "fairness_guardrail_triggered_step_rate": 0.0,
                        "avg_admitted_queue_wait_s": 0.1,
                        "p95_queue_wait_s": 0.2,
                        "max_queue_wait_s": 0.3,
                        "per_class_p95_queue_wait_s": {"latency": 0.2},
                    },
                }
            },
            "benchmark": {
                "derived_metrics": {
                    "prefix_cache": {
                        "request_count": 8.0,
                        "lookup_hit_rate": 0.5,
                        "materialized_hit_rate": 0.25,
                        "saved_prefill_tokens_per_request": 6.0,
                        "saved_prefill_tokens_per_materialized_hit": 24.0,
                        "lookup_cost_per_request": 4.5,
                        "lookup_candidates_per_request": 1.5,
                    },
                    "preemption": {
                        "step_count": 10.0,
                        "preempted_steps": 4.0,
                        "preempted_prefill_requests": 7.0,
                        "preempted_step_rate": 0.4,
                        "preempted_prefill_requests_per_step": 0.7,
                    },
                    "fairness": {
                        "step_count": 10.0,
                        "admitted_requests": 8.0,
                        "starvation_protected_steps": 2.0,
                        "fairness_guardrail_triggered_steps": 3.0,
                        "starvation_protected_step_rate": 0.2,
                        "fairness_guardrail_triggered_step_rate": 0.3,
                        "avg_admitted_queue_wait_s": 0.4,
                        "p95_queue_wait_s": 0.9,
                        "max_queue_wait_s": 1.2,
                        "per_class_p95_queue_wait_s": {
                            "latency": 0.8,
                            "background": 1.1,
                        },
                    },
                }
            },
        }
    )

    assert diffs == {
        "baseline_phase": "warmup",
        "target_phase": "benchmark",
        "prefix_cache": {
            "warmup": {
                "request_count": 1.0,
                "lookup_hit_rate": 0.0,
                "materialized_hit_rate": 0.0,
                "saved_prefill_tokens_per_request": 0.0,
                "saved_prefill_tokens_per_materialized_hit": 0.0,
                "lookup_cost_per_request": 2.0,
                "lookup_candidates_per_request": 1.0,
            },
            "benchmark": {
                "request_count": 8.0,
                "lookup_hit_rate": 0.5,
                "materialized_hit_rate": 0.25,
                "saved_prefill_tokens_per_request": 6.0,
                "saved_prefill_tokens_per_materialized_hit": 24.0,
                "lookup_cost_per_request": 4.5,
                "lookup_candidates_per_request": 1.5,
            },
            "benchmark_delta": {
                "request_count": 7.0,
                "lookup_hit_rate": 0.5,
                "materialized_hit_rate": 0.25,
                "saved_prefill_tokens_per_request": 6.0,
                "saved_prefill_tokens_per_materialized_hit": 24.0,
                "lookup_cost_per_request": 2.5,
                "lookup_candidates_per_request": 0.5,
            },
        },
        "preemption": {
            "warmup": {
                "step_count": 2.0,
                "preempted_steps": 0.0,
                "preempted_prefill_requests": 0.0,
                "preempted_step_rate": 0.0,
                "preempted_prefill_requests_per_step": 0.0,
            },
            "benchmark": {
                "step_count": 10.0,
                "preempted_steps": 4.0,
                "preempted_prefill_requests": 7.0,
                "preempted_step_rate": 0.4,
                "preempted_prefill_requests_per_step": 0.7,
            },
            "benchmark_delta": {
                "step_count": 8.0,
                "preempted_steps": 4.0,
                "preempted_prefill_requests": 7.0,
                "preempted_step_rate": 0.4,
                "preempted_prefill_requests_per_step": 0.7,
            },
        },
        "fairness": {
            "warmup": {
                "step_count": 2.0,
                "admitted_requests": 2.0,
                "starvation_protected_steps": 0.0,
                "fairness_guardrail_triggered_steps": 0.0,
                "starvation_protected_step_rate": 0.0,
                "fairness_guardrail_triggered_step_rate": 0.0,
                "avg_admitted_queue_wait_s": 0.1,
                "p95_queue_wait_s": 0.2,
                "max_queue_wait_s": 0.3,
                "per_class_p95_queue_wait_s": {"latency": 0.2},
            },
            "benchmark": {
                "step_count": 10.0,
                "admitted_requests": 8.0,
                "starvation_protected_steps": 2.0,
                "fairness_guardrail_triggered_steps": 3.0,
                "starvation_protected_step_rate": 0.2,
                "fairness_guardrail_triggered_step_rate": 0.3,
                "avg_admitted_queue_wait_s": 0.4,
                "p95_queue_wait_s": 0.9,
                "max_queue_wait_s": 1.2,
                "per_class_p95_queue_wait_s": {
                    "latency": 0.8,
                    "background": 1.1,
                },
            },
            "benchmark_delta": {
                "step_count": 8.0,
                "admitted_requests": 6.0,
                "starvation_protected_steps": 2.0,
                "fairness_guardrail_triggered_steps": 3.0,
                "starvation_protected_step_rate": 0.2,
                "fairness_guardrail_triggered_step_rate": 0.3,
                "avg_admitted_queue_wait_s": 0.30000000000000004,
                "p95_queue_wait_s": 0.7,
                "max_queue_wait_s": 0.8999999999999999,
                "per_class_p95_queue_wait_s": {
                    "background": 1.1,
                    "latency": 0.6000000000000001,
                },
            },
        },
    }


def test_format_runtime_phase_diff_summary_renders_high_signal_lines() -> None:
    lines = _format_runtime_phase_diff_summary(
        {
            "prefix_cache": {
                "benchmark_delta": {
                    "materialized_hit_rate": 0.25,
                    "saved_prefill_tokens_per_request": 6.0,
                    "lookup_cost_per_request": 2.5,
                }
            },
            "preemption": {
                "benchmark_delta": {
                    "preempted_step_rate": 0.4,
                    "preempted_prefill_requests_per_step": 0.7,
                }
            },
            "fairness": {
                "benchmark_delta": {
                    "fairness_guardrail_triggered_step_rate": 0.3,
                    "starvation_protected_step_rate": 0.2,
                    "p95_queue_wait_s": 0.7,
                    "per_class_p95_queue_wait_s": {
                        "background": 1.1,
                        "latency": 0.6,
                    },
                }
            },
        }
    )

    assert lines == [
        "  RUNTIME(prefix): mat_hit_rate_delta=+0.250, saved_prefill_tok_per_req_delta=+6.000, lookup_cost_per_req_delta=+2.500",
        "  RUNTIME(preempt): step_rate_delta=+0.400, prefills_per_step_delta=+0.700",
        "  RUNTIME(fair): guardrail_rate_delta=+0.300, starvation_rate_delta=+0.200, p95_queue_wait_delta=+0.700s",
        "  RUNTIME(fair,p95_by_class): background=+1.100s, latency=+0.600s",
    ]
