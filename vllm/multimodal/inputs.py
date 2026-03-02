# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypeAlias, Union, TypedDict
import torch

# --- 基础数据项定义 ---

class ImageItem(TypedDict):
    """单张图像输入"""
    image: Any 
    
class AudioItem(TypedDict):
    """音频输入"""
    audio: Any 
    sampling_rate: int

class VideoItem(TypedDict):
    """视频输入"""
    video: Any 

ModalityData: TypeAlias = Union[torch.Tensor, List[torch.Tensor], 
                                ImageItem, AudioItem, VideoItem]

MultiModalDataDict: TypeAlias = Mapping[str, Union[ModalityData, List[ModalityData]]]

# 为兼容性补全所有在 multimodal/__init__.py 或其他模块中引用的定义
MultiModalDataBuiltins: TypeAlias = Any
MultiModalUUIDDict: TypeAlias = Any
MultiModalPlaceholderDict: TypeAlias = Any
MultiModalKwargsItems: TypeAlias = Any
MultiModalKwargs: TypeAlias = Any
NestedTensors: TypeAlias = Any
class MultiModalFeatureSpec: pass
class MultiModalInputProcessor: pass
class MultiModalInputsBuilder: pass
class BaseMultiModalField: pass
class MultiModalBatchedField: pass
class MultiModalFieldElem: pass
class MultiModalFlatField: pass
class MultiModalKwargsItem: pass
class MultiModalSharedField: pass

# --- 引擎内部使用的输入封装 ---

@dataclass
class MultiModalInputs:
    """
    封装预处理后的多模态数据，直接传递给 ModelExecutor。
    """
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    multi_modal_data: Optional[MultiModalDataDict] = None
    mm_placeholders: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchedTensorInputs:
    """批处理后的多模态张量"""
    pixel_values: Optional[torch.Tensor] = None
    pixel_attention_mask: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    audio_values: Optional[torch.Tensor] = None
    audio_attention_mask: Optional[torch.Tensor] = None

# --- 字段与配置定义 ---

class MultiModalFieldConfig:
    def __init__(self, name: str, modality: str):
        self.name = name
        self.modality = modality

class MultiModalLimitSpec:
    def __init__(self, limit_per_prompt: Dict[str, int]):
        self.limit_per_prompt = limit_per_prompt
