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
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct  :class:`~vllm-multimodal.input.PlaceholderRange`
        for each multi-modal item.
        In contrast to the base class, this method always adds
        `return_mm_token_type_ids` to the processor data
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        vLLM language model class. Therefore, in order to return a language model vLLM
