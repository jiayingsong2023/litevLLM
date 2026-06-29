from __future__ import annotations

import pytest
import torch

from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashContextEstimate,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


class _FakeTokenizer:
    def __init__(self, eos_token_id: int = 0) -> None:
        self.eos_token_id = eos_token_id

    def encode(self, prompt: str) -> list[int]:
        return [ord(char) for char in prompt]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class _ReadyBackend:
    is_ready = True
    missing_kernels: tuple[str, ...] = ()
    _stats = {"cpu_token_sync_points": 0}

    def require_ready(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        del kv_rows, attn_sinks, token_idx
        assert query.is_cuda
        return query

    def compressed_attention(
        self,
        *,
        query: torch.Tensor,
        compressed_rows: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> torch.Tensor:
        del compressed_rows, selected_rows
        assert query.is_cuda
        return query

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        del gate_weight, up_weight, down_weight
        assert hidden.is_cuda
        return torch.zeros_like(hidden, dtype=torch.float32)

    def output_logits(self, **kwargs: torch.Tensor | int) -> torch.Tensor:
        tensor_args = [
            value for value in kwargs.values() if isinstance(value, torch.Tensor)
        ]
        assert tensor_args
        assert all(tensor.is_cuda for tensor in tensor_args)
        rows = kwargs["lm_head_values"]
        assert isinstance(rows, torch.Tensor)
        return torch.arange(rows.shape[0], dtype=torch.float32, device=rows.device)


class _FakeStore:
    def __init__(
        self,
        shape: DeepSeekV4FlashShape,
        *,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = type(
            "GGUF",
            (),
            {
                "metadata": (
                    {}
                    if eos_token_id is None
                    else {"tokenizer.ggml.eos_token_id": eos_token_id}
                )
            },
        )()
        self.bindings = type(
            "Bindings",
            (),
            {
                "token_embedding": object(),
            },
        )()


def _runtime_budget(context_length: int) -> DeepSeekV4FlashRuntimeBudget:
    return DeepSeekV4FlashRuntimeBudget(
        context=DeepSeekV4FlashContextEstimate(
            context_length=context_length,
            raw_kv_bytes=0,
            compressed_kv_bytes=0,
            scratch_bytes=0,
        ),
        model_mmap_bytes=1,
        resident_weight_bytes=0,
        expert_cache_bytes=0,
        uma_budget_bytes=1,
        min_system_headroom_bytes=0,
    )


def _fake_model(
    *,
    context_length: int = 16,
    vocab_size: int = 11,
    eos_token_id: int | None = None,
) -> DeepSeekV4FlashForCausalLM:
    shape = DeepSeekV4FlashShape(
        num_layers=0,
        hidden_size=32,
        vocab_size=vocab_size,
        head_dim=32,
    )
    return DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(  # type: ignore[arg-type]
            shape,
            eos_token_id=eos_token_id,
        ),
        runtime_budget=_runtime_budget(context_length),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )


def _make_engine(*, direct_runtime: bool = True) -> LiteEngine:
    engine = object.__new__(LiteEngine)
    engine.device = torch.device("cpu")
    engine.model = _fake_model(context_length=32, vocab_size=16)
    engine.tokenizer = _FakeTokenizer()
    engine._deepseek_v4_flash_direct = direct_runtime
    return engine


def _greedy_sampling_params(max_tokens: int = 3) -> SamplingParams:
    return SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )


def test_generate_deepseek_v4_flash_greedy_batched_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_engine()
    max_tokens = 3

    def fake_batched(
        input_ids_list: list[torch.Tensor],
        *,
        max_tokens: int,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for input_ids in input_ids_list:
            last = int(input_ids[-1].item())
            generated = [
                (last + offset) % engine.model.shape.vocab_size
                for offset in range(1, max_tokens + 1)
            ]
            outputs.append(
                torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            generated,
                            dtype=torch.long,
                            device=input_ids.device,
                        ),
                    ]
                )
            )
        return outputs

    monkeypatch.setattr(
        engine.model,
        "generate_greedy_kernel_batched",
        fake_batched,
    )

    request_ids = ["req-0", "req-1"]
    prompts = ["ab", "xyz"]
    sampling_params_list = [_greedy_sampling_params(max_tokens) for _ in prompts]

    outputs = engine.generate_deepseek_v4_flash_greedy_batched(
        request_ids=request_ids,
        prompts=prompts,
        sampling_params_list=sampling_params_list,
    )

    assert len(outputs) == len(prompts)
    for output, request_id, prompt in zip(outputs, request_ids, prompts, strict=True):
        assert isinstance(output, RequestOutput)
        assert output.request_id == request_id
        assert output.prompt == prompt
        assert output.finished is True
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        assert isinstance(completion, CompletionOutput)
        assert completion.index == 0

    # ``ab`` encodes to [97, 98]; the batched kernel appends (98+1..98+3)%16.
    assert outputs[0].prompt_token_ids == [97, 98]
    assert outputs[0].outputs[0].token_ids == [3, 4, 5]
    assert outputs[0].outputs[0].text == "".join(chr(t) for t in [3, 4, 5])

    # ``xyz`` encodes to [120, 121, 122]; generated ids start at (122+1)%16.
    assert outputs[1].prompt_token_ids == [120, 121, 122]
    assert outputs[1].outputs[0].token_ids == [11, 12, 13]


def test_generate_deepseek_v4_flash_greedy_batched_ignores_use_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_engine()

    def fake_batched(
        input_ids_list: list[torch.Tensor],
        *,
        max_tokens: int,
    ) -> list[torch.Tensor]:
        return [
            torch.cat(
                [
                    input_ids,
                    torch.zeros(
                        max_tokens,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                ]
            )
            for input_ids in input_ids_list
        ]

    monkeypatch.setattr(engine.model, "generate_greedy_kernel_batched", fake_batched)

    outputs = engine.generate_deepseek_v4_flash_greedy_batched(
        request_ids=["req-0"],
        prompts=["a"],
        sampling_params_list=[_greedy_sampling_params(max_tokens=2)],
        use_graph=True,
    )

    assert len(outputs) == 1
    assert outputs[0].finished is True


def test_generate_deepseek_v4_flash_greedy_batched_requires_direct_runtime() -> None:
    engine = _make_engine(direct_runtime=False)

    with pytest.raises(RuntimeError, match="direct runtime is not enabled"):
        engine.generate_deepseek_v4_flash_greedy_batched(
            request_ids=["req-0"],
            prompts=["hello"],
            sampling_params_list=[_greedy_sampling_params()],
        )


@pytest.mark.parametrize(
    "bad_params",
    [
        SamplingParams(max_tokens=0),
        SamplingParams(n=2, max_tokens=1),
        SamplingParams(temperature=0.5, max_tokens=1),
        SamplingParams(top_p=0.9, max_tokens=1),
        SamplingParams(top_k=5, max_tokens=1),
    ],
)
def test_generate_deepseek_v4_flash_greedy_batched_rejects_invalid_sampling_params(
    bad_params: SamplingParams,
) -> None:
    engine = _make_engine()

    with pytest.raises(ValueError):
        engine.generate_deepseek_v4_flash_greedy_batched(
            request_ids=["req-0"],
            prompts=["hello"],
            sampling_params_list=[bad_params],
        )


def test_generate_deepseek_v4_flash_greedy_batched_rejects_mismatched_lengths() -> None:
    engine = _make_engine()

    with pytest.raises(ValueError, match="same length"):
        engine.generate_deepseek_v4_flash_greedy_batched(
            request_ids=["req-0", "req-1"],
            prompts=["hello"],
            sampling_params_list=[_greedy_sampling_params()],
        )


def test_engine_batched_empty_prompts_returns_empty_list() -> None:
    engine = _make_engine()

    outputs = engine.generate_deepseek_v4_flash_greedy_batched(
        request_ids=[],
        prompts=[],
        sampling_params_list=[],
    )

    assert outputs == []
