from abc import ABC, abstractmethod
from typing import Any

from vllm.outputs import RequestOutput


class OutputProcessorStrategy(ABC):
    @abstractmethod
    def process_outputs(self, request_outputs: list[RequestOutput], **kwargs) -> Any:
        pass
