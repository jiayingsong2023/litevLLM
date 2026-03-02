# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
import torch
from vllm.multimodal.inputs import MultiModalInputs

_I = TypeVar("_I")

class InputProcessingContext: pass
class MultiModalInputContext: pass
class MultiModalProcessingMetadata: pass
class MultiModalProcessorContext: pass
class MultiModalProcessingInfo: pass
class PromptUpdate: pass

class BaseProcessingInfo:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.allowed_mm_limits: Dict[str, int] = {"image": 1, "audio": 0}

class BaseDummyInputsBuilder(Generic[_I]): pass

class BaseMultiModalProcessor(ABC, Generic[_I]):
    def __init__(self, info: _I, dummy_inputs: Any):
        self.info = info
        self.dummy_inputs = dummy_inputs

    @abstractmethod
    def apply(
        self,
        prompt: str,
        mm_items: Dict[str, Any],
        **kwargs
    ) -> MultiModalInputs:
        raise NotImplementedError

class ImageProcessor(BaseMultiModalProcessor):
    def apply(
        self,
        prompt: str,
        mm_items: Dict[str, Any],
        **kwargs
    ) -> MultiModalInputs:
        images = mm_items.get("image", [])
        if not isinstance(images, list):
            images = [images]
        pixel_values = torch.randn(len(images), 3, 224, 224, device="cuda")
        return MultiModalInputs(
            prompt=prompt,
            multi_modal_data={"image": pixel_values}
        )

class MultiModalInputProcessor:
    def __init__(self, registry: Any):
        self.registry = registry

    def process_inputs(self, prompt: str, mm_data: Dict[str, Any]) -> MultiModalInputs:
        if "image" in mm_data:
            processor = ImageProcessor(None, None)
            return processor.apply(prompt, mm_data)
        return MultiModalInputs(prompt=prompt)
