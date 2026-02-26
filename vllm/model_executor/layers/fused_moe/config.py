# SPDX-License-Identifier: Apache-2.0
class FusedMoEConfig:
    def __init__(self, num_experts: int, top_k: int, intermediate_size: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size

class FusedMoEParallelConfig:
    def __init__(self):
        self.tp_size = 1
        self.ep_size = 1