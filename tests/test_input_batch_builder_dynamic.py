# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def _make_builder(num_layers=2, num_blocks_per_seq=4):
    kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(8, 16, 2, 32)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))
    mgr = KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=[(None, None)] * num_layers,
        num_blocks_per_seq=num_blocks_per_seq,
        block_size=16,
        max_active_requests=2,
        block_allocator=BlockAllocator(8),
    )

    class Cfg:
        kv_type = "fp16"
        k_scale = 1.0
        v_scale = 1.0

    return InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=64,
        num_layers=num_layers,
        kv_block_manager=mgr,
        inf_config=Cfg(),
        stack_per_layer_carries=lambda *_: None,
        split_per_layer_carries=lambda *_: None,
    ), mgr


def test_prefill_slot_mapping_dynamic_blocks():
    builder, mgr = _make_builder()
    req = RequestState(
        request_id="r0",
        prompt="hi",
        input_ids=list(range(20)),
        sampling_params=SamplingParams(),
        slot_idx=0,
    )
    mgr.ensure_blocks("r0", 20)

    class Scheduler:
        def get_request(self, rid):
            return req

    curr_input, positions, attn_metadata, _, _ = builder.build_prefill(
        ["r0"], Scheduler(), chunk_len=20
    )
    slot_mapping = attn_metadata["slot_mapping"]
    # First token in first block, 20th token starts second block.
    assert slot_mapping[0].item() == mgr._request_blocks["r0"][0] * 16
    assert slot_mapping[16].item() == mgr._request_blocks["r0"][1] * 16
