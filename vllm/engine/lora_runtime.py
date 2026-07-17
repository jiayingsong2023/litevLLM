# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.lora.request import LoRARequest


def _stable_lora_int_id(name: str) -> int:
    return max(1, abs(hash(name)))


@dataclass
class LoRAAdapterRecord:
    request: LoRARequest
    active_requests: int = 0
    total_requests: int = 0


@dataclass
class LoRARuntimeRegistry:
    _adapters: dict[str, LoRAAdapterRecord] = field(default_factory=dict)

    def register_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str | None = None,
        lora_int_id: int | None = None,
    ) -> LoRARequest:
        name = str(lora_name or "").strip()
        if not name:
            raise ValueError("LoRA adapter name must be non-empty")
        record = self._adapters.get(name)
        resolved_int_id = int(lora_int_id or _stable_lora_int_id(name))
        if record is None:
            request = LoRARequest(
                lora_name=name,
                lora_int_id=resolved_int_id,
                lora_path=lora_path,
            )
            self._adapters[name] = LoRAAdapterRecord(request=request)
            return request
        if lora_path is not None:
            record.request.lora_path = lora_path
        if lora_int_id is not None:
            record.request.lora_int_id = resolved_int_id
        return record.request

    def unregister_adapter(self, lora_name: str) -> bool:
        name = str(lora_name or "").strip()
        if not name:
            return False
        record = self._adapters.get(name)
        if record is None:
            return False
        if record.active_requests > 0:
            raise ValueError(
                f"cannot unregister LoRA adapter '{name}' while {record.active_requests} requests are active"
            )
        del self._adapters[name]
        return True

    def resolve_adapter(
        self,
        *,
        lora_id: str | None = None,
        lora_request: Any | None = None,
    ) -> LoRARequest | None:
        if lora_request is None and not lora_id:
            return None

        if lora_request is not None:
            name = str(getattr(lora_request, "lora_name", "") or lora_id or "").strip()
            if not name:
                raise ValueError("LoRA request is missing lora_name")
            return self.register_adapter(
                lora_name=name,
                lora_path=getattr(lora_request, "lora_path", None),
                lora_int_id=getattr(lora_request, "lora_int_id", None),
            )

        name = str(lora_id or "").strip()
        record = self._adapters.get(name)
        if record is None:
            raise ValueError(f"LoRA adapter '{name}' is not registered")
        return record.request

    def on_request_added(self, lora_name: str | None) -> None:
        if not lora_name:
            return
        record = self._adapters.get(str(lora_name))
        if record is None:
            raise ValueError(f"LoRA adapter '{lora_name}' is not registered")
        record.active_requests += 1
        record.total_requests += 1

    def on_request_removed(self, lora_name: str | None) -> None:
        if not lora_name:
            return
        record = self._adapters.get(str(lora_name))
        if record is None:
            return
        record.active_requests = max(0, record.active_requests - 1)

    def stats(self) -> dict[str, Any]:
        return {
            "registered_adapters": len(self._adapters),
            "active_adapters": sum(
                1 for record in self._adapters.values() if record.active_requests > 0
            ),
            "active_requests": sum(
                record.active_requests for record in self._adapters.values()
            ),
            "total_routed_requests": sum(
                record.total_requests for record in self._adapters.values()
            ),
            "adapters": {
                name: {
                    "lora_int_id": record.request.lora_int_id,
                    "lora_path": record.request.lora_path,
                    "active_requests": record.active_requests,
                    "total_requests": record.total_requests,
                }
                for name, record in sorted(self._adapters.items())
            },
        }
