from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.engine.lite_engine import LiteEngine
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
        self.calls: list[tuple[list[int], int, str]] = []
        self.prepared: tuple[int, str] | None = None

    def prepare_for_serving(
        self,
        *,
        context_length: int,
        device: torch.device,
    ) -> dict[str, int | None]:
        self.prepared = (context_length, device.type)
        return {}

    def generate_greedy_kernel(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int = 1,
        use_graph: bool = False,
    ) -> torch.Tensor:
        self.calls.append(
            (
                input_ids.detach().cpu().tolist(),
                max_tokens,
                input_ids.device.type,
                use_graph,
            )
        )
        assert max_tokens == 2
        return torch.tensor(
            [3, 7, 11, 12],
            dtype=torch.long,
            device=input_ids.device,
        )


def _deepseek_engine() -> LiteEngine:
    engine = LiteEngine.__new__(LiteEngine)
    engine.model = _FakeDeepSeekModel()
    engine.tokenizer = _FakeTokenizer()
    engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine.direct_runtime = DeepSeekV4FlashDirectRuntime(
        model=engine.model,
        model_config=SimpleNamespace(max_model_len=4096),
        runtime_config=SimpleNamespace(queue_timeout_s=30.0),
        tokenizer=engine.tokenizer,
        device=engine.device,
        observer=None,
    )
    return engine


def test_lite_engine_deepseek_direct_greedy_uses_kernel_generate() -> None:
    engine = _deepseek_engine()

    output = engine.direct_runtime.generate(
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
    assert engine.model.calls == [([3, 7], 2, engine.device.type, False)]


def test_lite_engine_deepseek_direct_rejects_non_greedy_sampling() -> None:
    engine = _deepseek_engine()

    with pytest.raises(ValueError, match="greedy"):
        engine.direct_runtime.generate(
            request_id="req-1",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1, temperature=0.7),
        )


def test_lite_engine_deepseek_direct_rejects_batch_outputs() -> None:
    engine = _deepseek_engine()

    with pytest.raises(ValueError, match="n=1"):
        engine.direct_runtime.generate(
            request_id="req-1",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0, n=2),
        )


def test_lite_engine_marks_deepseek_direct_runtime_without_kv_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeDeepSeekModel()
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(model="deepseek.gguf"),
        runtime_config=SimpleNamespace(policy_mode="auto"),
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.RuntimeConfig.from_vllm_config",
        lambda _cfg: cfg.runtime_config,
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.get_model_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            runtime_policy=lambda *_a, **_k: SimpleNamespace(
                tuning_env_overrides={},
                model_policy={},
                kernel_policy={},
            ),
            build_direct_runtime=lambda **kwargs: DeepSeekV4FlashDirectRuntime(
                **kwargs
            ),
        ),
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.get_model",
        lambda vllm_config: fake_model,
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.LiteEngine._apply_runtime_model_policy",
        lambda self: None,
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.LiteEngine._install_runtime_policy_on_runtime_config",
        lambda self, policy: None,
    )
    monkeypatch.setattr(
        "vllm.engine.lite_engine.LiteEngine._install_tuning_configs_for_model",
        lambda self, policy: None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DeepSeek direct runtime must not allocate generic KV cache")
        ),
    )

    engine = LiteEngine(cfg)  # type: ignore[arg-type]

    assert engine.model is fake_model
    assert engine.direct_runtime is not None
    assert not hasattr(engine, "_deepseek_v4_flash_direct")
    assert engine.model.prepared == (engine.max_model_len, "cuda")
    tokenizer = _FakeTokenizer()
    engine.set_tokenizer(tokenizer)
    assert engine.direct_runtime.tokenizer is tokenizer
