# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass
class _DummyGroup:
    world_size: int = 1
    group_name: str = "single"


class All2AllManagerBase:
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.dp_world_size = 1
        self.tp_group = _DummyGroup()
        self.cpu_group = _DummyGroup()
        self.internode = False
        self.workspace_tensor = None
        self.prepare_workspace_tensor = None

    def ensure_alltoall_workspace_initialized(self) -> bool:
        if self.workspace_tensor is None:
            self.workspace_tensor = torch.empty(1, device="cpu")
        if self.prepare_workspace_tensor is None:
            self.prepare_workspace_tensor = torch.empty(1, device="cpu")
        return True

    def get_handle(self, _args):
        return None


class DeviceCommunicatorStub:
    def __init__(self):
        self.all2all_manager = All2AllManagerBase()
