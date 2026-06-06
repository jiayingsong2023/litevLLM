import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.deepseek_v4_flash.moe import (
    topk_router_reference,
)


def _sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x).sqrt()


def test_topk_router_reference_returns_deepseek_v4_scaled_topk_weights() -> None:
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
    scores = _sqrtsoftplus(logits)
    expected_expert_ids = torch.topk(scores, k=2, sorted=True).indices
    expected_weights = scores.gather(0, expected_expert_ids)
    expected_weights = expected_weights / (expected_weights.sum() + 1e-20)
    expected_weights = expected_weights * 1.5

    torch.testing.assert_close(expert_ids, expected_expert_ids)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.5))
    assert expert_ids.dtype == torch.int64
    assert weights.dtype == torch.float32


def test_topk_router_reference_softmax_mode_keeps_selected_weight_semantics() -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    expert_ids, weights = topk_router_reference(
        hidden,
        router_weight,
        top_k=2,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
    )

    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    scores = torch.softmax(logits, dim=0)
    expected_weights, expected_expert_ids = torch.topk(
        scores,
        k=2,
        sorted=True,
    )
    expected_weights = expected_weights / (expected_weights.sum() + 1e-20)

    torch.testing.assert_close(expert_ids, expected_expert_ids)
    torch.testing.assert_close(weights, expected_weights)


def test_topk_router_reference_correction_bias_only_affects_selected_experts() -> None:
    hidden = torch.tensor([1.0], dtype=torch.float32)
    router_weight = torch.tensor([[3.0], [1.0], [0.0]], dtype=torch.float32)
    correction_bias = torch.tensor([-10.0, 3.0, 5.0], dtype=torch.float16)

    expert_ids, weights = topk_router_reference(
        hidden,
        router_weight,
        top_k=2,
        correction_bias=correction_bias,
    )

    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    scores = _sqrtsoftplus(logits)
    expected_expert_ids = torch.topk(
        scores + correction_bias.to(torch.float32),
        k=2,
        sorted=True,
    ).indices
    expected_weights = scores.gather(0, expected_expert_ids)
    expected_weights = expected_weights / (expected_weights.sum() + 1e-20)
    expected_weights = expected_weights * 1.5

    torch.testing.assert_close(expert_ids, expected_expert_ids)
    torch.testing.assert_close(weights, expected_weights)
    assert expert_ids.tolist() == [2, 1]


def test_topk_router_reference_weights_are_positive_and_scaled() -> None:
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
    torch.testing.assert_close(weights.sum(), torch.tensor(1.5))


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
            torch.empty(0),
            torch.ones((3, 0)),
            "hidden must contain at least one element",
        ),
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
        (
            torch.ones(2),
            torch.ones((0, 2)),
            "router_weight must contain at least one expert row",
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
