from abc import ABC, abstractmethod
from typing import List, Any
from vllm.outputs import RequestOutput

class OutputProcessorStrategy(ABC):
    @abstractmethod
    def process_outputs(self, request_outputs: List[RequestOutput], **kwargs) -> Any:
        pass
