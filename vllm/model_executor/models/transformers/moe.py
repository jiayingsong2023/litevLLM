# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

    # --8<-- [end:transformers_fused_moe]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._topk_ids: torch.Tensor = None

        def custom_routing_function(hidden_states, gating_output, topk, renormalize):
            topk_weights = gating_output
            topk_ids = self._topk_ids
            # Handle all gather in expert parallel
            if topk_ids.size(0) != hidden_states.size(0):
                dp_metadata = get_forward_context().dp_metadata
                sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
                is_sp = self.is_sequence_parallel
                dist_group = get_ep_group() if is_sp else get_dp_group()
                assert sizes[dist_group.rank_in_group] == topk_ids.shape[0]
                (topk_ids,) = dist_group.all_gatherv([topk_ids], 0, sizes)
            return topk_weights, topk_ids

        self.custom_routing_function = custom_routing_function

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.ops.vllm.transformers_moe_forward(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            self.layer_name,
        )

def transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
        Params for weights, fp8 weight scales, fp8 activation scales
        (param_name, weight_name, expert_id, shard_id)
        text_config = self.text_config

        # Positional arguments
        num_experts = self.model_config.get_num_experts()
        top_k = getattr_iter(text_config, ["num_experts_per_tok", "top_k"], None)
        assert top_k is not None
        hidden_size = text_config.hidden_size
        intermediate_size = getattr_iter(
            text_config, ["moe_intermediate_size", "intermediate_size"], None
        )
        assert intermediate_size is not None

        # If there are shared experts, the results are
        # reduced after mlp.forward() not inside FusedMoE
        num_shared_experts = getattr_iter(
            text_config,
            [
                "n_shared_experts",  # DeepSeek, Docs, GLM
                "moe_num_shared_experts",  # Aria, Ernie
            ],
            0,
        )
        reduce_results = num_shared_experts == 0

        def add_all_reduce(mlp: nn.Module):
