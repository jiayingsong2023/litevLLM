# SPDX-License-Identifier: Apache-2.0
from typing import Generic, TypeVar

_I = TypeVar("_I")

class MultiModalProcessor: pass
class MultiModalProcessorContext: pass
class MultiModalProcessingMetadata: pass
class BaseDummyInputsBuilder(Generic[_I]): pass
class MultiModalInputsBuilder: pass
class BaseMultiModalProcessor(Generic[_I]): pass
class MultiModalDataDict: pass
class MultiModalRegistry: pass
class BaseProcessingInfo: pass
class MultiModalProcessingInfo: pass
class PromptUpdate: pass
class InputProcessingContext: pass
class MultiModalInputContext: pass
