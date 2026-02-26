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
        super().__init_subclass__(*args, **kwargs)
        hf_to_vllm_mapper = WeightsMapper()
        for base in cls.__mro__:
            if base_hf_to_vllm_mapper := getattr(base, "hf_to_vllm_mapper", None):
                hf_to_vllm_mapper |= base_hf_to_vllm_mapper
        cls.hf_to_vllm_mapper = hf_to_vllm_mapper

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__()
        logger.info("Using Transformers modeling backend.")

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

        # Attrs for weight loading (see self.load_weights)
        self.skip_prefixes: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []

        # Attrs for Eagle3 (see self.set_aux_hidden_state_layers)
        self._target_class: type[nn.Module] = nn.Module
        self._output_aux_hidden_states_kwargs: dict[str, bool] = {}
        Apply the model's pipeline parallelization plan.

        Currently, this replaces:

        - `nn.Linear` with vLLM's tensor parallel linear classes
        - `*RMSNorm` with vLLM's `RMSNorm`
        Create `Attention` instances to inform KV cache allocation.
        If a `parameter` is on the `meta` device, then its parent
        `module` is the original module created by:

        ```python
        with torch.device("meta"):
            self.model: "PreTrainedModel" = AutoModel.from_config(...)
        ```
