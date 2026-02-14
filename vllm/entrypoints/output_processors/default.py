from typing import List, Any
from vllm.outputs import RequestOutput
from vllm.entrypoints.output_processors.abstract import OutputProcessorStrategy

class DefaultOutputProcessor(OutputProcessorStrategy):
    """
    Default output processor that returns the list of RequestOutput objects as-is.
    This maintains the standard vLLM behavior.
    """
    def process_outputs(self, request_outputs: List[RequestOutput], **kwargs) -> List[RequestOutput]:
        return request_outputs
