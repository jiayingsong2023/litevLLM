import torch
from typing import List, Any, Dict
from vllm.outputs import RequestOutput
from vllm.entrypoints.output_processors.abstract import OutputProcessorStrategy

class VerlOutputProcessor(OutputProcessorStrategy):
    """
    Verl-specific output processor that vectorizes the output processing
    to reduce CPU overhead, specifically designed for RLHF workflows.
    """
    def process_outputs(self, request_outputs: List[RequestOutput], **kwargs) -> Dict[str, torch.Tensor]:
        if not request_outputs:
            return {}

        token_ids_list = []
        all_cum_logprobs = []
        max_len = 0
        
        # Collect data and find max length for padding
        for req in request_outputs:
            # We assume n=1 for RLHF training loops usually
            if not req.outputs:
                continue
            
            output = req.outputs[0]
            ids = output.token_ids
            token_ids_list.append(ids)
            all_cum_logprobs.append(output.cumulative_logprob)
            
            if len(ids) > max_len:
                max_len = len(ids)

        # Pad in Python to avoid overhead of creating many small tensors
        # pad_value=0 is a placeholder (often pad_token_id)
        # This allows us to create one single tensor in one go, which is much faster
        # than torch.nn.utils.rnn.pad_sequence with a list of tensors created in a loop.
        # TODO: Pass pad_token_id via kwargs if needed.
        
        padded_ids_list = []
        for ids in token_ids_list:
            # list concatenation is fast in Python
            padded_ids_list.append(ids + [0] * (max_len - len(ids)))

        if padded_ids_list:
            # Creating a tensor from a rectangular list of lists is efficient
            token_ids_tensor = torch.tensor(padded_ids_list, dtype=torch.long)
        else:
            token_ids_tensor = torch.empty((0, 0), dtype=torch.long)

        # Cumulative logprobs are just a 1D tensor
        cum_logprobs_tensor = torch.tensor(all_cum_logprobs, dtype=torch.float32)

        return {
            "token_ids": token_ids_tensor,
            "cumulative_logprob": cum_logprobs_tensor,
            # Add other fields as necessary for Verl
        }
