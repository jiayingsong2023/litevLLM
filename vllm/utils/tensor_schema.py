# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

class TensorShape:
    def __init__(
        self,
        *dims: int | str,
        dynamic_dims: set[str] | None = None,
    ) -> None:
        super().__init__()

        self.dims = dims
        self.dynamic_dims = dynamic_dims if dynamic_dims else set()

    def resolve(self, **bindings: int) -> tuple[int | str, ...]:
        resolved = list[int | str]()
        for dim in self.dims:
            if isinstance(dim, str) and dim in bindings:
                resolved.append(bindings[dim])
            else:
                resolved.append(dim)
        return tuple(resolved)

    def __str__(self) -> str:
        if isinstance(value, (int, float)):
            return ()  # Scalar
        if isinstance(value, torch.Tensor):
            return value.shape

        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{field_name}{self._fmt_indexer(leading_idxs)} is not "
                f"one of the expected types: int, float, Tensor, list, tuple. "
                f"Got: {type(value)}"
            )

        if len(value) == 0:
            raise ValueError(
                f"{field_name}{self._fmt_indexer(leading_idxs)} is an empty sequence"
            )

        # Ensure all tensors in the list have the same
        # shape, besides dynamic dimensions
        for i, v in enumerate(value):
            shape = self._validate_field(
                v,
                field_name,
                expected_shape[1:],
                dynamic_dims,
                leading_idxs=leading_idxs + (i,),
            )

            if i == 0:
                first_shape = shape
            elif not self._match_shape_with_dynamic(
                shape,
                first_shape,
                expected_shape,
                dynamic_dims,
            ):
                raise ValueError(
                    f"{field_name}{self._fmt_indexer(leading_idxs)} "
                    f"contains inconsistent shapes: {first_shape} "
                    f"(index 0) vs {shape} (index {i})"
                )

        # Treat the list as a stacked tensor:
        # shape = (len(list), *tensor.shape)
        return (len(value),) + first_shape

    def _validate_tensor_shape_expected(
        self,
        actual_shape: tuple[int, ...],
        expected_shape: tuple[int | str, ...],
        field_name: str,
        shape_env: dict[str, int],
        dynamic_dims: set[str],
    ) -> None:
        logger.debug("Shapes in %s:", self.__class__.__name__)
        type_hints = get_type_hints(self.__class__, include_extras=True)

        for field_name, field_type in type_hints.items():
            if get_origin(field_type) is not None:
                args = get_args(field_type)
                for arg in args:
                    if isinstance(arg, TensorShape):
                        logger.debug("  %s: %s", field_name, str(arg))
