# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, cast, overload

import torch
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import direct_register_custom_op

if not has_helion():
    raise ImportError(
        "register module requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
from helion.autotuner.base_search import BaseAutotuner
from helion.runtime.config import Config
from helion.runtime.settings import default_autotuner_fn

logger = init_logger(__name__)

vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa

def validate_helion_settings(
    helion_settings: "helion.Settings | None", op_name: str
) -> None:

    def __init__(
        self,
        args: tuple[Any, ...],
        config_selector: Callable[[tuple[Any, ...]], Config],
    ):
        self.args = args
        self.config_selector = config_selector

    def autotune(self, *, skip_cache: bool = False) -> Config:
        return self.config_selector(self.args)

class ConfiguredHelionKernel:
        Create a key computer function derived from the config picker.

        The returned function receives kernel arguments unpacked (*args) to match
        Helion's key signature (called as self._key_fn(*args)).

    def __init__(
        self,
        raw_kernel_func: Callable,
        op_name: str,
        fake_impl: Callable,
        helion_settings: "helion.Settings | None" = None,
    ):
        # Validate helion_settings doesn't conflict with our custom autotuner
        validate_helion_settings(helion_settings, op_name)

        self.raw_kernel_func = raw_kernel_func
        self.op_name = op_name
        self._fake_impl = fake_impl
        self.helion_settings = helion_settings
        self._config_picker: (
            Callable[[tuple[Any, ...], list[str]], str | None] | None
        ) = None

    def __call__(self, *args, **kwargs):
        configured_op = self.get_configured_op()
        return configured_op(*args, **kwargs)

    def register_config_picker(
        self, picker_func: Callable[[tuple[Any, ...], list[str]], str | None]
    ) -> Callable[[tuple[Any, ...], list[str]], str | None]:
        self._config_picker = picker_func
        return picker_func

    def get_configured_op(self) -> Any:
        assert self._config_picker is not None, (
            f"No config picker registered for kernel '{self.op_name}'. "
            f"Use @{self.op_name}.register_config_picker to register one."
        )

        if hasattr(torch.ops.vllm_helion, self.op_name):
            logger.debug("Op vllm_helion::%s already registered", self.op_name)
            return getattr(torch.ops.vllm_helion, self.op_name)

        configured_kernel = ConfiguredHelionKernel(
            op_name=self.op_name,
            config_picker=self._config_picker,
            raw_kernel_func=self.raw_kernel_func,
            helion_settings=self.helion_settings,
        )

        logger.info("Registering op: vllm_helion::%s", self.op_name)
        direct_register_custom_op(
            op_name=self.op_name,
            op_func=configured_kernel._decorated_kernel,  # Register decorated kernel
            # TODO(gmagogsfm): Implement automatic mutation/aliasing detection
            # for Helion kernels.
            mutates_args=None,
            fake_impl=self._fake_impl,
            target_lib=vllm_helion_lib,
        )
        return getattr(torch.ops.vllm_helion, self.op_name)

# Global registry for tracking all registered HelionKernelWrapper instances
_REGISTERED_KERNELS: dict[str, HelionKernelWrapper] = {}

def get_registered_kernels() -> dict[str, HelionKernelWrapper]:
    return _REGISTERED_KERNELS.copy()

def get_kernel_by_name(kernel_name: str) -> HelionKernelWrapper | None:
    return _REGISTERED_KERNELS.get(kernel_name)

def infer_fake_impl(
    kernel_func: Callable,
    helion_settings: "helion.Settings | None" = None,
) -> Callable:
    def helion_fake_kernel(*args, **kwargs):
        kernel_kwargs = {}
        if helion_settings:
            kernel_kwargs.update(helion_settings.to_dict())

        temp_decorated_kernel = helion.kernel(**kernel_kwargs)(kernel_func)

        # Bind with args to get config_spec, then get a valid default config
        bound = temp_decorated_kernel.bind(args)
        default_config = bound.config_spec.default_config()
        compiled_runner = bound.compile_config(default_config)

        return compiled_runner(*args, **kwargs, _launcher=lambda *a, **kw: None)

    return helion_fake_kernel

# Overloads are necessary for proper mypy type inference.
# Without overloads, the union return type HelionKernelWrapper | Callable[...]
# causes mypy to complain about missing attributes when tests do:
#   wrapper = register_kernel(func)  # Should return HelionKernelWrapper
#   wrapper._fake_impl  # mypy error: "Callable has no attribute _fake_impl"
# The overloads tell mypy the exact return type based on the argument pattern.
@overload
def register_kernel(
    op_name_or_func: Callable,
    *,
    fake_impl: Callable | None = None,
    helion_settings: "helion.Settings | None" = None,
) -> HelionKernelWrapper: ...

@overload
def register_kernel(
    op_name_or_func: str | None = None,
    *,
    fake_impl: Callable | None = None,
    helion_settings: "helion.Settings | None" = None,
) -> Callable[[Callable], HelionKernelWrapper]: ...

def register_kernel(
    op_name_or_func: str | Callable | None = None,
    *,
    fake_impl: Callable | None = None,
    helion_settings: "helion.Settings | None" = None,
) -> HelionKernelWrapper | Callable[[Callable], HelionKernelWrapper]:

    def decorator(kernel_func: Callable) -> HelionKernelWrapper:
        op_name = op_name_or_func if isinstance(op_name_or_func, str) else None
        final_op_name = op_name if op_name else kernel_func.__name__

        if final_op_name in _REGISTERED_KERNELS:
            raise ValueError(
                f"Helion kernel '{final_op_name}' is already registered. "
                f"Use a different op_name or check for duplicate registrations."
            )

        final_fake_impl = fake_impl
        if final_fake_impl is None:
            final_fake_impl = infer_fake_impl(kernel_func, helion_settings)
            logger.debug(
                "Auto-generated fake_impl for Helion kernel '%s'",
                kernel_func.__name__,
            )

        kernel_wrapper = HelionKernelWrapper(
            raw_kernel_func=kernel_func,
            op_name=final_op_name,
            fake_impl=final_fake_impl,
            helion_settings=helion_settings,
        )

        _REGISTERED_KERNELS[final_op_name] = kernel_wrapper

        logger.info(
            "Registered Helion kernel '%s' as HelionKernelWrapper",
            kernel_func.__name__,
        )

        return kernel_wrapper

    if callable(op_name_or_func) and not isinstance(op_name_or_func, str):
        # Bare decorator usage: @register_kernel
        return decorator(op_name_or_func)
    else:
        # Decorator with arguments: @register_kernel(...)
        return decorator
