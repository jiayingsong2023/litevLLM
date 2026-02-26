# SPDX-License-Identifier: Apache-2.0
class SchedulerConfig:
    def __init__(self, max_num_batched_tokens: int, max_num_seqs: int, max_model_len: int):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len