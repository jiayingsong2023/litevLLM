from vllm.entrypoints.output_processors.abstract import OutputProcessorStrategy
from vllm.outputs import RequestOutput


class DefaultOutputProcessor(OutputProcessorStrategy):
    def process_outputs(
        self, request_outputs: list[RequestOutput], **kwargs
    ) -> list[RequestOutput]:
        return request_outputs
