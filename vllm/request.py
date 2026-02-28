# SPDX-License-Identifier: Apache-2.0
import enum
from typing import Optional, List, Any

class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3
    FINISHED_ABORTED = 4
    FINISHED_IGNORED = 5
    WAITING_FOR_FSM = 6
    WAITING_FOR_STREAMING_REQ = 7

    @staticmethod
    def is_finished(status):
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_IGNORED,
        ]

class Request:
    def __init__(self, request_id: str, prompt_token_ids: List[int], **kwargs):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(prompt_token_ids)
        self.num_output_tokens = 0
        self.num_output_placeholders = 0
        self.num_cached_tokens = -1
        self.num_preemptions = 0
        self.arrival_time = 0.0
        self.priority = 0
        self.sampling_params = None
        self.pooling_params = None
        self.lora_request = None
        self.has_encoder_inputs = False
        self.use_structured_output = False
        self.spec_token_ids = []
        self._all_token_ids = prompt_token_ids.copy()
        self.stop_reason = None
        self.trace_headers = None
        self.num_nans_in_logits = 0
    
    @property
    def num_tokens(self):
        return len(self._all_token_ids)
        
    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)
    
    @property
    def all_token_ids(self):
        return self._all_token_ids

    def is_finished(self):
        return RequestStatus.is_finished(self.status)
    
    def record_event(self, *args, **kwargs): pass
    def take_events(self): return []
    def get_finished_reason(self): return None

class StreamingUpdate:
    @classmethod
    def from_request(cls, request): return None
