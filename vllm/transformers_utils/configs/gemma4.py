# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class _Gemma4ProportionalRopeValidationMixin:
    def _validate_proportional_rope_parameters(
        self,
        rope_parameters: dict[str, Any],
        ignore_keys: set[str] | None = None,
    ) -> None:
        required_keys = {"rope_type", "rope_theta"}
        optional_keys = {"partial_rotary_factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type,
            received_keys,
            required_keys,
            optional_keys,
            ignore_keys=ignore_keys,
        )

        rope_theta = rope_parameters["rope_theta"]
        if rope_theta is None or not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
            raise ValueError(
                f"`rope_parameters`'s rope_theta field must be a positive number, got {rope_theta}"
            )

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is not None and (
            not isinstance(partial_rotary_factor, (int, float))
            or partial_rotary_factor <= 0
            or partial_rotary_factor > 1.0
        ):
            raise ValueError(
                "`rope_parameters`'s partial_rotary_factor field must be in (0, 1], "
                f"got {partial_rotary_factor}"
            )


class Gemma4TextConfig(_Gemma4ProportionalRopeValidationMixin, PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Gemma4VisionConfig(PretrainedConfig):
    model_type = "gemma4_vision"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Gemma4Config(_Gemma4ProportionalRopeValidationMixin, PretrainedConfig):
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | Gemma4TextConfig | None = None,
        vision_config: dict[str, Any] | Gemma4VisionConfig | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        self.text_config = text_config

        if vision_config is None:
            self.vision_config = None
        elif isinstance(vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if architectures is None:
            architectures = ["Gemma4ForConditionalGeneration"]

        super().__init__(architectures=architectures, **kwargs)

        # Lite runtime reads most decoder fields from the top-level hf_config.
        for k, v in vars(self.text_config).items():
            if k.startswith("_"):
                continue
            if not hasattr(self, k) or getattr(self, k) is None:
                setattr(self, k, v)

        if getattr(self, "layer_types", None) is None:
            self.layer_types = getattr(self.text_config, "layer_types", None)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        return self.text_config


def build_fallback_hf_config(config_dict: dict[str, Any]) -> PretrainedConfig:
    model_type = str(config_dict.get("model_type", "") or "").lower()
    text_model_type = str(
        ((config_dict.get("text_config") or {}).get("model_type", "")) or ""
    ).lower()

    if model_type == "gemma4" or text_model_type.startswith("gemma4"):
        data = dict(config_dict)
        if "dtype" not in data and "torch_dtype" in data:
            data["dtype"] = data["torch_dtype"]
        return Gemma4Config(**data)

    cfg = PretrainedConfig()
    if "dtype" not in config_dict and "torch_dtype" in config_dict:
        setattr(cfg, "dtype", config_dict["torch_dtype"])
    for k, v in config_dict.items():
        if k == "torch_dtype":
            continue
        if k != "text_config":
            setattr(cfg, k, v)
    if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
        for k, v in config_dict["text_config"].items():
            setattr(cfg, k, v)
        cfg.layer_types = config_dict["text_config"].get("layer_types", [])
    return cfg


for model_type, config_cls in (
    ("gemma4", Gemma4Config),
    ("gemma4_text", Gemma4TextConfig),
    ("gemma4_vision", Gemma4VisionConfig),
):
    try:
        AutoConfig.register(model_type, config_cls)
    except ValueError:
        pass
