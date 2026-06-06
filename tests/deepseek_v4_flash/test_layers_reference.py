import pytest
import torch
from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.models.deepseek_v4_flash.layers import (
    deepseek_linear_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)


def test_deepseek_linear_reference_multiplies_row_major_weight() -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float16)
    weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16)

    out = deepseek_linear_reference(hidden, weight)

    torch.testing.assert_close(out, torch.tensor([11.0, 17.0]))
    assert out.dtype == torch.float32


def test_deepseek_linear_reference_rejects_non_1d_hidden() -> None:
    hidden = torch.ones((1, 2), dtype=torch.float32)
    weight = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="hidden must be 1-D"):
        deepseek_linear_reference(hidden, weight)


def test_deepseek_linear_reference_rejects_non_2d_weight() -> None:
    hidden = torch.ones(2, dtype=torch.float32)
    weight = torch.ones(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="weight must be 2-D"):
        deepseek_linear_reference(hidden, weight)


def test_deepseek_linear_reference_rejects_shape_mismatch() -> None:
    hidden = torch.ones(3, dtype=torch.float32)
    weight = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="weight columns must match hidden size"):
        deepseek_linear_reference(hidden, weight)


def test_weight_store_tensor_payload_returns_exact_tensor_bytes(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    payload = b"\x01\x02\x03\x04"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(0, 0),
        tensor_dims={
            "token_embd.weight": (1,),
            "blk.0.attn_q.weight": (1,),
        },
        tensor_payloads={
            "token_embd.weight": payload,
            "blk.0.attn_q.weight": b"\x05\x06\x07\x08",
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        view = store.tensor_payload(store.bindings.token_embedding)
        try:
            assert view.tobytes() == payload
        finally:
            view.release()
