# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Any

class CompletionOutput:
    """The output data of one completion output of a request."""
    def __init__(self, index: int, text: str, token_ids: List[int], cumulative_logprob: float, **kwargs):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob

class RequestOutput:
    """The output data of a request to the LLM."""
    def __init__(self, request_id: str, prompt: str, prompt_token_ids: List[int], outputs: List[CompletionOutput], finished: bool, **kwargs):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished

class PoolingOutput: pass
class PoolingRequestOutput: pass
