from __future__ import annotations

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _ensure_hyper_connection_streams_batched,
    _hyper_connection_post_cuda_batched,
    _hyper_connection_pre_cuda_batched,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashQuantizedExpertPayload,
)


class _FakeStore:
    def __init__(self) -> None:
        self.matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        return torch.zeros(1, 1)

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        return DeepSeekV4FlashQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=0,
            rows=1,
            columns=1,
            payload=memoryview(b""),
        )

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        return self.matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.vectors[(tensor.name, dtype)].clone()

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        return memoryview(b"")


def _tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_F16,
        offset=0,
        nbytes=0,
    )


def _make_hc_tensors(
    *,
    hc_mult: int,
    hidden_size: int,
    name_prefix: str = "hc",
) -> tuple[DeepSeekV4FlashHyperConnectionTensors, _FakeStore]:
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    store = _FakeStore()
    fn_t = _tensor(f"{name_prefix}_fn.weight", (flat_size, mix_count))
    base_t = _tensor(f"{name_prefix}_base.weight", (mix_count,))
    scale_t = _tensor(f"{name_prefix}_scale.weight", (3,))
    store.matrices[fn_t.name] = torch.randn(
        flat_size, mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )
    return (
        DeepSeekV4FlashHyperConnectionTensors(fn=fn_t, base=base_t, scale=scale_t),
        store,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ensure_hyper_connection_streams_batched_expands_2d() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    hidden = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    streams = _ensure_hyper_connection_streams_batched(
        hidden, stager=stager, hyper_connection=hc
    )
    assert streams.shape == (3, 2, 8)
    torch.testing.assert_close(streams[0, 0], hidden[0])
    torch.testing.assert_close(streams[0, 1], hidden[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hyper_connection_pre_post_batched_matches_loop() -> None:
    from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
        _ensure_hyper_connection_streams,
        _hyper_connection_post_cuda,
        _hyper_connection_pre_cuda,
    )

    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    hidden = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    streams = _ensure_hyper_connection_streams_batched(
        hidden, stager=stager, hyper_connection=hc
    )

    state_b = _hyper_connection_pre_cuda_batched(
        streams, stager=stager, hyper_connection=hc
    )

    singles = [
        _hyper_connection_pre_cuda(
            _ensure_hyper_connection_streams(
                hidden[b], stager=stager, hyper_connection=hc
            ),
            stager=stager,
            hyper_connection=hc,
        )
        for b in range(hidden.shape[0])
    ]

    torch.testing.assert_close(state_b.mixed, torch.stack([s.mixed for s in singles]))
    torch.testing.assert_close(state_b.post, torch.stack([s.post for s in singles]))
    torch.testing.assert_close(
        state_b.combine, torch.stack([s.combine for s in singles])
    )

    update = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    out_b = _hyper_connection_post_cuda_batched(update, streams, state_b)
    singles_out = torch.stack(
        [
            _hyper_connection_post_cuda(
                update[b],
                _ensure_hyper_connection_streams(
                    hidden[b], stager=stager, hyper_connection=hc
                ),
                singles[b],
            )
            for b in range(hidden.shape[0])
        ]
    )
    torch.testing.assert_close(out_b, singles_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ensure_hyper_connection_streams_batched_preserves_3d() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    streams = torch.randn(3, 2, 8, dtype=torch.float32, device="cuda")
    out = _ensure_hyper_connection_streams_batched(
        streams, stager=stager, hyper_connection=hc
    )
    assert out is not streams  # clones
    torch.testing.assert_close(out, streams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hyper_connection_post_batched_shape_validation() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    streams = torch.randn(3, 2, 8, dtype=torch.float32, device="cuda")
    state = _hyper_connection_pre_cuda_batched(
        streams, stager=stager, hyper_connection=hc
    )
    bad_output = torch.randn(3, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="batched mHC output shape must be"):
        _hyper_connection_post_cuda_batched(bad_output, streams, state)
