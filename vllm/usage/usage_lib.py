# SPDX-License-Identifier: Apache-2.0
from enum import Enum

class UsageContext(Enum):
    UNKNOWN = 0
    LLM_CLASS = 1
    ENGINE_CONTEXT = 2
    API_SERVER = 3
    OPENAI_API_SERVER = 4
