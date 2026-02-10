# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .layer_utils import replace_parameter, update_tensor_inplace
from .misc_utils import quant_noise, tensor_force_quant

__all__ = ["update_tensor_inplace", "replace_parameter", "quant_noise", "tensor_force_quant"]
