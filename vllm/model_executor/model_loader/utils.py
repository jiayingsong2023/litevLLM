# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model

    msg = (
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(**kwargs)
        record_metadata_for_reloading(model)

    return model

def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)

    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)

@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched

_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
