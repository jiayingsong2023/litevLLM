from abc import ABC, abstractmethod
from typing import List, Any
from vllm.outputs import RequestOutput

class OutputProcessorStrategy(ABC):
    """
    Abstract base class for output processing strategies.
    Allows customizing how the LLM engine outputs are processed and returned.
    """
    @abstractmethod
    def process_outputs(self, request_outputs: List[RequestOutput], **kwargs) -> Any:
        """
        Process the list of RequestOutput objects into the desired format.
        """
        pass
