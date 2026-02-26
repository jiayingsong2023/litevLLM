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
    A context manager under which models are initialized with all
    parameters on the specified device. However buffers are not
    initialized on specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear: `nn.Linear` to be replaced.
        style: Tensor parallel style of the new linear, e.g. "colwise".
        quant_config: Quantization config for the new linear.
    Returns:
        The new linear.

    Args:
        conv: `nn.Conv2d` or `nn.Conv3d` to be replaced.
    Returns:
        The new `Conv2dLayer` or `Conv3dLayer`. If the conv module is not supported,
        returns the original conv module.

    This method assumes:
    - Weight is stored as `weight`.
    - Epsilon is stored as `eps` or `variance_epsilon`.
    - `with_scale` indicates whether the layer has a weight (Gemma3n only).
    - `var_hidden_size` is only ever used for Intern vision encoder in vLLM
    and Transformers doesn't appear to have the same concept.
    Callable to be passed to `@support_torch_compile`'s `enable_if` argument.

    Defaults to `True` but is disabled in the following situations:

    - The model uses dynamic rope scaling.
