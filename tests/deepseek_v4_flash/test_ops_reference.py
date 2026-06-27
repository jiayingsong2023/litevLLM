import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.ops import (
    factorized_linear_reference,
    rms_norm_reference,
    silu_gate_reference,
)


def test_rms_norm_reference_matches_formula() -> None:
    hidden = torch.tensor([3.0, 4.0])
    weight = torch.tensor([2.0, 1.0])

    out = rms_norm_reference(hidden, weight, eps=0.0)

    expected = hidden * torch.rsqrt(hidden.pow(2).mean()) * weight
    torch.testing.assert_close(out, expected)


def test_factorized_linear_reference_applies_a_then_b() -> None:
    hidden = torch.tensor([1.0, 2.0])
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])

    out = factorized_linear_reference(hidden, a, b)

    torch.testing.assert_close(out, torch.tensor([4.0, 4.0]))


def test_silu_gate_reference_applies_silu_to_gate_then_multiplies_up() -> None:
    gate = torch.tensor([0.0, 1.0], dtype=torch.float16)
    up = torch.tensor([2.0, 3.0], dtype=torch.bfloat16)

    out = silu_gate_reference(gate, up)

    expected = torch.nn.functional.silu(gate.to(torch.float32)) * up.to(
        torch.float32
    )
    torch.testing.assert_close(out, expected)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    ("hidden", "weight", "message"),
    [
        (torch.ones((1, 2)), torch.ones(2), "hidden must be 1-D"),
        (torch.ones(2), torch.ones((1, 2)), "weight must be 1-D"),
        (torch.ones(3), torch.ones(2), "weight length must match hidden size"),
    ],
)
def test_rms_norm_reference_validates_shapes(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rms_norm_reference(hidden, weight)


def test_silu_gate_reference_rejects_shape_mismatch() -> None:
    gate = torch.ones(2)
    up = torch.ones(3)

    with pytest.raises(ValueError, match="gate and up shapes must match"):
        silu_gate_reference(gate, up)


@pytest.mark.parametrize(
    ("hidden", "a", "b", "message"),
    [
        (
            torch.ones((1, 2)),
            torch.ones((3, 2)),
            torch.ones((2, 3)),
            "hidden must be 1-D",
        ),
        (
            torch.ones(2),
            torch.ones(3),
            torch.ones((2, 3)),
            "a must be 2-D",
        ),
        (
            torch.ones(2),
            torch.ones((3, 2)),
            torch.ones(3),
            "b must be 2-D",
        ),
        (
            torch.ones(3),
            torch.ones((3, 2)),
            torch.ones((2, 3)),
            "a columns must match hidden size",
        ),
        (
            torch.ones(2),
            torch.ones((4, 2)),
            torch.ones((2, 3)),
            "b columns must match a rows",
        ),
    ],
)
def test_factorized_linear_reference_validates_shapes(
    hidden: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factorized_linear_reference(hidden, a, b)
