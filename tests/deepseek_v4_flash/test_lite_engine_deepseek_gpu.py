from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.adapters.base import ModelCapabilities, RuntimeModelPolicy
from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.engine.executor_result import TokenDecodeResult, TokenPrefillResult
from vllm.engine.lite_engine import LiteEngine
from vllm.engine.runtime_config import RuntimeConfig
from vllm.sampling_params import SamplingParams


class _CustomTokenizer:
    eos_token_id = 99

    def encode(self, prompt: str) -> list[int]:
        return [int(part) for part in prompt.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


class _CustomKV:
    block_size = 16
    num_blocks_per_seq = 8
    num_layers = 1

    def __init__(self) -> None:
        self.active: set[str] = set()

    def ensure_blocks_for_requests(
        self,
        request_ids: list[str],
        token_counts: list[int],
    ) -> None:
        del token_counts
        self.active.update(request_ids)

    def free_request_blocks(self, request_id: str) -> None:
        self.active.discard(request_id)

    def stats(self) -> dict[str, int]:
        return {"active_requests": len(self.active)}

    def capture_prefix_entry(self, **kwargs):
        raise AssertionError("DeepSeek custom path should not capture prefix cache")


class _CustomPrefill:
    def execute(self, request_ids, scheduler, chunk_len):
        prefilled = []
        for request_id in request_ids:
            req = scheduler.get_request(request_id)
            assert chunk_len == len(req.input_ids)
            prefilled.append(len(req.input_ids))
        return TokenPrefillResult(
            torch.tensor([10] * len(request_ids)),
            prefilled,
            [True] * len(request_ids),
        )


class _CustomDecode:
    def execute_sync_fast(self, request_ids, scheduler):
        return self.execute_batch(request_ids, scheduler)

    def execute_batch(self, request_ids, scheduler):
        del scheduler
        return TokenDecodeResult(torch.tensor([11] * len(request_ids)))


class _CustomAdapter:
    model_type = "deepseek_v4_flash"

    def __init__(self, kv: _CustomKV) -> None:
        self.kv = kv

    def named_modules(self):
        return []

    def runtime_policy(self, model_config, runtime_config):
        del model_config, runtime_config
        return RuntimeModelPolicy(
            model_policy={"runtime_budget_bytes": 175},
            kernel_policy={},
        )

    def install_tuning_config(self, tuning_env):
        del tuning_env

    def detect(self, model, model_config):
        del model, model_config
        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=1,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=8,
            max_model_len=64,
            supports_moe=True,
            supports_fp8_kv=False,
            supports_int4_kv=False,
            supports_paged_prefill=False,
            preferred_kv_dtype="deepseek_v4_compressed",
            supports_chunked_prefill=False,
        )

    def build_executors(self, **kwargs):
        del kwargs
        return CustomRuntimeComponents(
            prefill_executor=_CustomPrefill(),
            decode_executor=_CustomDecode(),
            kv_block_manager=self.kv,
        )

    def estimate_kv_bytes(self, *, max_active_requests, context_length):
        del context_length
        return int(max_active_requests) * 100

    def estimate_staging_bytes(self, *, max_active_requests):
        return int(max_active_requests) * 50

    def validate_request(self, **kwargs):
        del kwargs


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        model_path="deepseek.fake",
        tokenizer_path="deepseek.fake",
        dtype="float16",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        block_size=16,
        kv_cache_dtype="fp16",
        kv_max_model_len=64,
        kv_max_active_requests=2,
        fusion_level=0,
        policy_mode="accuracy",
        enable_decode_priority=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        min_prefill_chunk_size=1,
        max_prefill_chunk_size=None,
        prefill_sla_ttft_ms=0.0,
    )


def _custom_engine(monkeypatch: pytest.MonkeyPatch) -> tuple[LiteEngine, _CustomKV]:
    kv = _CustomKV()
    adapter = _CustomAdapter(kv)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(model="deepseek.fake", hf_config=None),
        runtime_config=_runtime_config(),
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.get_model_adapter",
        lambda *_args, **_kwargs: adapter,
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.get_model",
        lambda vllm_config: adapter,
    )
    monkeypatch.setattr("vllm.engine.lite_engine.get_total_gpu_memory_gb", lambda: 16.0)
    monkeypatch.setattr(
        "vllm.engine.runtime_planner.get_total_gpu_memory_gb",
        lambda: 16.0,
    )
    original_empty = torch.empty

    def fake_empty(*args, **kwargs):
        kwargs["device"] = torch.device("cpu")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    engine = LiteEngine(cfg)  # type: ignore[arg-type]
    engine.set_tokenizer(_CustomTokenizer())
    return engine, kv


@pytest.mark.asyncio
async def test_lite_engine_deepseek_custom_runtime_completes_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, kv = _custom_engine(monkeypatch)

    assert engine.step_scheduler.prefill_chunk_size == engine.max_model_len
    assert engine.step_scheduler.prefill_microbatch_size == 1
    assert engine.input_batch_builder is None

    engine.add_request(
        "req-1",
        "1 2 3",
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    engine.step()
    engine.step()

    outputs = []
    async for output in engine.get_request_stream("req-1"):
        outputs.append(output)

    assert outputs[-1].finished is True
    assert outputs[-1].outputs[0].token_ids == [10]
    assert kv.stats()["active_requests"] == 0


def test_lite_engine_deepseek_custom_runtime_rejects_over_budget_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _kv = _custom_engine(monkeypatch)

    with pytest.raises(Exception, match="full-prefill"):
        engine.add_request(
            "req-long",
            " ".join(str(i) for i in range(17)),
            SamplingParams(max_tokens=1, temperature=0.0),
        )


def test_lite_engine_deepseek_custom_runtime_abort_frees_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, kv = _custom_engine(monkeypatch)
    engine.add_request(
        "req-1",
        "1 2 3",
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    engine.step()
    engine.step()

    assert kv.stats()["active_requests"] == 1
    engine.abort_request("req-1")

    assert kv.stats()["active_requests"] == 0


def test_lite_engine_deepseek_admission_cap_uses_model_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _kv = _custom_engine(monkeypatch)

    assert engine.max_active_requests == 1
    assert engine.scheduler.max_active_requests == 1
    engine.add_request(
        "req-1",
        "1 2 3",
        SamplingParams(max_tokens=2, temperature=0.0),
    )

    with pytest.raises(Exception, match="request admission cap reached"):
        engine.add_request(
            "req-2",
            "4 5 6",
            SamplingParams(max_tokens=2, temperature=0.0),
        )
