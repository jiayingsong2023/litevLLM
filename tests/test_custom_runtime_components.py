from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.engine.multimodal_processor import NullMultiModalProcessor


def test_custom_runtime_components_can_carry_null_multimodal_processor() -> None:
    processor = NullMultiModalProcessor()
    components = CustomRuntimeComponents(
        prefill_executor=object(),
        decode_executor=object(),
        kv_block_manager=object(),
        multimodal_processor=processor,
    )

    assert components.multimodal_processor is processor
    assert processor.prepare_request(SimpleNamespace(multi_modal_data=None)) is None
    assert processor.build_prefill_inputs([]) == {}
    assert processor.get_multimodal_embeddings({}) is None
    assert processor.stats() == {}
    processor.reset_stats()



def test_backend_accepts_token_prefill_and_decode_results() -> None:
    from types import SimpleNamespace

    import torch

    from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
    from vllm.engine.executor_result import TokenDecodeResult, TokenPrefillResult
    from vllm.engine.request_state import RequestState
    from vllm.sampling_params import SamplingParams

    request = RequestState(
        request_id="req-1",
        prompt="hi",
        input_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    events: list[str] = []

    class Scheduler:
        def get_request(self, request_id: str) -> RequestState:
            assert request_id == "req-1"
            return request

        def transition_to_decode(self, request_id: str) -> None:
            events.append(f"decode:{request_id}")

        def publish_output(self, request_id: str, output) -> None:
            events.append(f"publish:{request_id}:{output.token}")

        def free_request(self, request_id: str):
            events.append(f"free:{request_id}")
            return request

    class Prefill:
        def execute(self, request_ids, scheduler, chunk_len):
            assert request_ids == ["req-1"]
            assert chunk_len == 2
            return TokenPrefillResult(torch.tensor([10]), [2], [True])

    class Decode:
        def execute_sync_fast(self, request_ids, scheduler):
            return self.execute_batch(request_ids, scheduler)

        def execute_batch(self, request_ids, scheduler):
            assert request_ids == ["req-1"]
            return TokenDecodeResult(torch.tensor([11]))

    class Output:
        def finalize_step(self, request_id: str, req: RequestState, token: int):
            return SimpleNamespace(request_id=request_id, token=token)

    backend = LiteSingleGpuBackend(
        scheduler=Scheduler(),
        observer=SimpleNamespace(
            on_prefill_executed=lambda *_args: events.append("prefill_obs"),
            on_decode_executed=lambda *_args: events.append("decode_obs"),
            on_first_token=lambda *_args: None,
        ),
        prefill_executor=Prefill(),
        decode_executor=Decode(),
        sampling_driver=SimpleNamespace(completion_eos_ids=lambda _req: set()),
        output_coordinator=Output(),
        kv_block_manager=SimpleNamespace(
            block_size=16,
            num_blocks_per_seq=8,
            num_layers=1,
            free_request_blocks=lambda request_id: events.append(f"kv:{request_id}"),
        ),
    )

    results = []
    backend.run_prefills(
        SimpleNamespace(prefills=SimpleNamespace(request_ids=["req-1"], chunk_len=2)),
        results,
    )
    results.extend(backend.decode_step_sync(["req-1"]))
    backend.run_decodes(
        SimpleNamespace(decodes=SimpleNamespace(request_ids=["req-1"])),
        results,
    )

    assert request.seq_len == 4
    assert request.generated_ids == [10, 11, 11]
    assert events == [
        "decode:req-1",
        "publish:req-1:10",
        "prefill_obs",
        "publish:req-1:11",
        "publish:req-1:11",
        "decode_obs",
    ]
    assert [r.token for r in results] == [10, 11, 11]
