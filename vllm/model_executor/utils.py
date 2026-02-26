# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    Replace a parameter of a layer while maintaining the ability to reload the weight.
    Called within implementations of the `process_weights_after_loading` method.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_data: New data of the new parameter, or None to set the parameter to None
