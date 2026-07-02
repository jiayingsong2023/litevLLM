# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for SamplingDriver.sample_batch_tokens."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling_driver import SamplingDriver
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 2


class NoOpOutputProcessor:
    def apply_context_bias(
        self,
        logits,
        generated_ids,
        sampling_params,
        bias_token_ids,
        is_capital_question,
    ):
        return logits


@dataclass
class Config:
    batch_size: int
    vocab_size: int
    temperature: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float


def make_requests(config: Config) -> list[RequestState]:
    sp = SamplingParams(
        temperature=config.temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=config.repetition_penalty,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        ignore_eos=False,
        min_tokens=5,
    )
    rng = torch.Generator(device="cuda")
    requests = []
    for i in range(config.batch_size):
        generated_ids = [(i + j) % config.vocab_size for j in range(50)]
        requests.append(
            RequestState(
                request_id=f"r{i}",
                prompt="",
                input_ids=[1, 2, 3],
                sampling_params=sp,
                generated_ids=generated_ids,
                rng=rng,
            )
        )
    return requests


def bench(label: str, config: Config, warmup: int = 10, reps: int = 50) -> float:
    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    driver = SamplingDriver(FakeTokenizer(), None, policies)
    requests = make_requests(config)
    for _ in range(warmup):
        logits = torch.randn(
            config.batch_size, config.vocab_size, device="cuda", dtype=torch.float16
        )
        driver.sample_batch_tokens(logits, requests)
    times = []
    for _ in range(reps):
        logits = torch.randn(
            config.batch_size, config.vocab_size, device="cuda", dtype=torch.float16
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        driver.sample_batch_tokens(logits, requests)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    avg_ms = avg * 1e3
    print(
        f"{label:20s} bs={config.batch_size:3d} "
        f"vocab={config.vocab_size:6d} avg={avg_ms:7.3f}ms"
    )
    return avg


def main() -> None:
    for vocab_size in [32000, 151936]:
        print(f"\n--- vocab_size={vocab_size} ---")
        for bs in [1, 2, 4, 8, 16]:
            vectorized = bench(
                "penalties+sample", Config(bs, vocab_size, 0.7, 1.1, 0.1, 0.1)
            )
            if bs in (1, 8, 16):
                os.environ["FASTINFERENCE_USE_LEGACY_SAMPLING"] = "1"
                legacy = bench(
                    "legacy penalties+sample",
                    Config(bs, vocab_size, 0.7, 1.1, 0.1, 0.1),
                )
                os.environ["FASTINFERENCE_USE_LEGACY_SAMPLING"] = "0"
                ratio = legacy / vectorized if vectorized > 0 else float("inf")
                print(
                    f"speedup             bs={bs:3d} "
                    f"vocab={vocab_size:6d} {ratio:6.2f}x"
                )
        for bs in [1, 8, 16]:
            bench("greedy", Config(bs, vocab_size, 0.0, 1.0, 0.0, 0.0))


if __name__ == "__main__":
    main()
