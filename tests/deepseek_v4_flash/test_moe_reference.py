import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.moe import (
    topk_router_reference,
)


def test_topk_router_reference_returns_sorted_topk_experts() -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float16)
    router_weight = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )

    expert_ids, weights = topk_router_reference(hidden, router_weight, top_k=2)

    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    expected_weights, expected_expert_ids = torch.topk(
        torch.softmax(logits, dim=0),
        k=2,
        sorted=True,
    )
    torch.testing.assert_close(expert_ids, expected_expert_ids)
    torch.testing.assert_close(weights, expected_weights)
    assert expert_ids.dtype == torch.int64
    assert weights.dtype == torch.float32


def test_topk_router_reference_weights_are_positive_normalized_probabilities() -> None:
    hidden = torch.tensor([0.5, -1.0], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float16,
    )

    _expert_ids, weights = topk_router_reference(hidden, router_weight, top_k=3)

    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("top_k", [0, 4])
def test_topk_router_reference_validates_top_k(top_k: int) -> None:
    hidden = torch.ones(2)
    router_weight = torch.ones((3, 2))

    with pytest.raises(ValueError, match="top_k must be > 0 and <= number of experts"):
        topk_router_reference(hidden, router_weight, top_k=top_k)


@pytest.mark.parametrize(
    ("hidden", "router_weight", "message"),
    [
        (
            torch.ones((1, 2)),
            torch.ones((3, 2)),
            "hidden must be 1-D",
        ),
        (
            torch.ones(2),
            torch.ones(2),
            "router_weight must be 2-D",
        ),
        (
            torch.ones(3),
            torch.ones((2, 2)),
            "router_weight columns must match hidden size",
        ),
    ],
)
def test_topk_router_reference_validates_shapes(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        topk_router_reference(hidden, router_weight, top_k=1)
