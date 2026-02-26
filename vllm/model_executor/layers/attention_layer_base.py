# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        pass
