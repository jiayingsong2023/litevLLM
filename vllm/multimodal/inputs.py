# SPDX-License-Identifier: Apache-2.0
from typing import Any, Mapping, TypeAlias

class BaseMultiModalField: pass
class MultiModalBatchedField: pass
class MultiModalFieldConfig: pass
class MultiModalFieldElem: pass
class MultiModalFlatField: pass
class MultiModalKwargsItem: pass
class MultiModalKwargsItems: pass
class MultiModalSharedField: pass
class NestedTensors: pass
class BatchedTensorInputs: pass
class MultiModalKwargs: pass
class MultiModalDataBuiltins: pass

ModalityData: TypeAlias = Any
MultiModalDataDict: TypeAlias = Mapping[str, Any]
MultiModalPlaceholderDict: TypeAlias = Any
MultiModalUUIDDict: TypeAlias = Any
class MultiModalFeatureSpec: pass
class MultiModalInputProcessor: pass
class MultiModalLimitSpec: pass
class MultiModalInputs: pass
