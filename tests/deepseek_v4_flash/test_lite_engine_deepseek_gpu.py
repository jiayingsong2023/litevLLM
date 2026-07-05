from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.adapters.base import ModelCapabilities, RuntimeModelPolicy
from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.engine.executor_result import TokenDecodeResult, TokenPrefillResult
from vllm.engine.lite_engine import LiteEngine
from vllm.engine.runtime_config import RuntimeConfig
from vllm.model_executor.models.deepseek_v4_flash.direct_runtime import (
    DeepSeekV4FlashDirectRuntime,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.sampling_params import SamplingParams


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "hello"
        return [3, 7]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        assert ids == [11, 12]
        assert skip_special_tokens is True
        return "world"


class _FakeDeepSeekModel(DeepSeekV4FlashForCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[int] | None, int, str | None]] = []
        self.prepared: tuple[int, str] | None = None

    def prepare_for_serving(
        self,
        *,
        context_length: int,
        device: torch.device,
    ) -> dict[str, int | None]:
        self.prepared = (context_length, device.type)
        return {}

    def prefill_greedy_kernel(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
    ) -> object:
        self.calls.append(
            (
                "prefill",
                input_ids.detach().cpu().tolist(),
                max_tokens,
                input_ids.device.type,
            )
        )
        return {"device": input_ids.device, "input_ids": input_ids}

    def decode_greedy_kernel(
        self,
        session: object,
        *,
        max_tokens: int,
        use_graph: bool = False,
    ) -> torch.Tensor:
        sess = session if isinstance(session, dict) else {}
        self.calls.append(("decode", None, max_tokens, str(use_graph)))
        assert max_tokens == 2
        return torch.tensor(
            [3, 7, 11, 12],
            dtype=torch.long,
            device=sess["device"],
        )

    def generate_greedy_kernel(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int = 1,
        use_graph: bool = False,
    ) -> torch.Tensor:
        raise AssertionError("direct runtime should use explicit prefill/decode stages")


def _deepseek_direct_runtime() -> tuple[
    DeepSeekV4FlashDirectRuntime,
    _FakeDeepSeekModel,
]:
    model = _FakeDeepSeekModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DeepSeekV4FlashDirectRuntime(
            model=model,
            model_config=SimpleNamespace(max_model_len=4096),
            runtime_config=SimpleNamespace(queue_timeout_s=30.0),
            tokenizer=_FakeTokenizer(),
            device=device,
            observer=None,
        ),
        model,
    )


def test_lite_engine_deepseek_direct_greedy_uses_kernel_generate() -> None:
    runtime, model = _deepseek_direct_runtime()

    output = runtime.generate(
        request_id="req-1",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
    )

    assert output.request_id == "req-1"
    assert output.prompt == "hello"
    assert output.prompt_token_ids == [3, 7]
    assert output.finished is True
    assert output.outputs[0].text == "world"
    assert output.outputs[0].token_ids == [11, 12]
    assert model.calls == [
        ("prefill", [3, 7], 2, runtime.device.type),
        ("decode", None, 2, "False"),
    ]


def test_lite_engine_deepseek_direct_rejects_non_greedy_sampling() -> None:
    runtime, _model = _deepseek_direct_runtime()

    with pytest.raises(ValueError, match="greedy"):
        runtime.generate(
            request_id="req-1",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1, temperature=0.7),
        )


def test_lite_engine_deepseek_direct_rejects_batch_outputs() -> None:
    runtime, _model = _deepseek_direct_runtime()

    with pytest.raises(ValueError, match="n=1"):
        runtime.generate(
            request_id="req-1",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0, n=2),
        )


def test_deepseek_adapter_does_not_build_direct_runtime() -> None:
    from vllm.adapters.deepseek_v4_flash import DeepSeekV4FlashAdapter

    adapter = DeepSeekV4FlashAdapter()

    assert (
        adapter.build_direct_runtime(
            model=object(),
            model_config=SimpleNamespace(max_model_len=4096),
            runtime_config=SimpleNamespace(queue_timeout_s=30.0),
            tokenizer=_FakeTokenizer(),
            device=torch.device("cpu"),
            observer=None,
        )
        is None
    )


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

    def runtime_policy(self, model_config, runtime_config):
        del model_config, runtime_config
        return RuntimeModelPolicy(model_policy={}, kernel_policy={})

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
        lambda vllm_config: SimpleNamespace(),
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
