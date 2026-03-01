# SPDX-License-Identifier: Apache-2.0
from enum import Enum

class EngineCoreEventType(Enum):
    QUEUED = 0
    SCHEDULED = 1
    PREEMPTED = 2

class EngineCoreOutput:
    def __init__(self, **kwargs):
        pass

class EngineCoreOutputs:
    def __init__(self, **kwargs):
        self.outputs = kwargs.get('outputs', [])
        self.scheduler_stats = None
        self.finished_requests = set()
