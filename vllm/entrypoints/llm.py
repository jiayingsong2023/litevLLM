# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Union
from vllm.inputs import PromptInput
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput

class LLM:
    def __init__(self, model: str, **kwargs):
        self.model = model
        
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        **kwargs,
    ) -> List[RequestOutput]:
        return []
