# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.lora.manager import LoRAManager
from vllm.lora.weights import LoRALayerWeights
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.sampling_params import SamplingParams


def _stack(req_dicts: list[RequestState], num_layers: int, key: str):
    del req_dicts, key
    return [None for _ in range(num_layers)]


def _split(stacked, req_dicts: list[RequestState], key: str) -> None:
    del stacked, req_dicts, key


def _make_builder(
    max_active_requests: int = 2,
) -> tuple[InputBatchBuilder, KVBlockManager]:
    kv_block_manager = KVBlockManager(
        kv_caches=[],
        kv_scale_caches=[],
        num_blocks_per_seq=2,
        block_size=2,
        max_active_requests=max_active_requests,
        block_allocator=BlockAllocator(num_total_blocks=8),
    )
    builder = InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=8,
        num_layers=1,
        kv_block_manager=kv_block_manager,
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )
    return builder, kv_block_manager


class _TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 2)
        self.proj = LiteLinear(2, 2, bias=False, prefix="proj")
        self.proj.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1] = torch.tensor([2.0, 3.0])
            self.embed.weight[2] = torch.tensor([5.0, 7.0])
            self.embed.weight[3] = torch.tensor([11.0, 13.0])
            self.embed.weight[4] = torch.tensor([17.0, 19.0])
        manager = LoRAManager()
        manager.add_adapter_weights(
            "adapter-a",
            {
                "proj": LoRALayerWeights(
                    lora_name="adapter-a",
                    rank=1,
                    alpha=1,
                    lora_a=torch.tensor([[1.0], [0.0]]),
                    lora_b=torch.tensor([[10.0, 20.0]]),
                )
            },
        )
        manager.add_adapter_weights(
            "adapter-b",
            {
                "proj": LoRALayerWeights(
                    lora_name="adapter-b",
                    rank=1,
                    alpha=1,
                    lora_a=torch.tensor([[0.0], [1.0]]),
                    lora_b=torch.tensor([[30.0, 40.0]]),
                )
            },
        )
        manager.bind_to_model(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: dict[str, Any],
        *,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        del positions, kv_caches, attn_metadata
        return self.proj(self.embed(input_ids), lora_mapping)


def test_mixed_lora_prefill_executor_applies_per_request_adapters() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.add_request(
        "r1",
        RequestState(
            request_id="r1",
            prompt="",
            slot_idx=0,
            is_prefill=True,
            seq_len=0,
            input_ids=[1],
            linear_attn_carry=[None],
            linear_conv_carry=[None],
            lora_id="adapter-a",
            sampling_params=SamplingParams(),
        ),
    )
    scheduler.add_request(
        "r2",
        RequestState(
            request_id="r2",
            prompt="",
            slot_idx=1,
            is_prefill=True,
            seq_len=0,
            input_ids=[2],
            linear_attn_carry=[None],
            linear_conv_carry=[None],
            lora_id="adapter-b",
            sampling_params=SamplingParams(),
        ),
    )
    builder, kv_block_manager = _make_builder()
    kv_block_manager.ensure_blocks("r1", 1)
    kv_block_manager.ensure_blocks("r2", 1)

    logits, _, _ = PrefillExecutor(
        model=_TinyLoRAModel(),
        input_batch_builder=builder,
        kv_caches=[],
    ).execute(["r1", "r2"], scheduler, chunk_len=1)

    assert torch.allclose(
        logits,
        torch.tensor(
            [
                [[22.0, 43.0]],
                [[215.0, 287.0]],
            ]
        ),
    )


def test_mixed_lora_decode_executor_keeps_base_rows_unchanged() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.add_request(
        "r1",
        RequestState(
            request_id="r1",
            prompt="",
            slot_idx=0,
            is_prefill=False,
            seq_len=1,
            input_ids=[1],
            generated_ids=[3],
            linear_attn_carry=[None],
            linear_conv_carry=[None],
            lora_id="adapter-a",
            sampling_params=SamplingParams(),
        ),
    )
    scheduler.add_request(
        "r2",
        RequestState(
            request_id="r2",
            prompt="",
            slot_idx=1,
            is_prefill=False,
            seq_len=1,
            input_ids=[2],
            generated_ids=[4],
            linear_attn_carry=[None],
            linear_conv_carry=[None],
            lora_id=None,
            sampling_params=SamplingParams(),
        ),
    )
    builder, kv_block_manager = _make_builder()
    kv_block_manager.ensure_blocks("r1", 2)
    kv_block_manager.ensure_blocks("r2", 2)

    logits, _ = DecodeExecutor(
        model=_TinyLoRAModel(),
        input_batch_builder=builder,
        kv_caches=[],
        fast_input_ids=torch.empty((2, 1), dtype=torch.long),
        fast_positions=torch.empty((2, 1), dtype=torch.long),
        fast_slot_mapping=torch.empty((2,), dtype=torch.long),
        fast_seq_lens=torch.empty((2,), dtype=torch.int32),
    ).execute_batch(["r1", "r2"], scheduler)

    assert torch.allclose(
        logits,
        torch.tensor(
            [
                [[121.0, 233.0]],
                [[17.0, 19.0]],
            ]
        ),
    )
