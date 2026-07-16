from types import SimpleNamespace

from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.adapters.deepseek_v4_flash import DeepSeekV4FlashAdapter
from vllm.adapters.registry import get_model_adapter
from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.model_executor.models.deepseek_v4_flash.executors import (
    DeepSeekDecodeExecutor,
    DeepSeekPrefillExecutor,
)
from vllm.model_executor.models.deepseek_v4_flash.kv_lifecycle import (
    DeepSeekKVLifecycleAdapter,
)


def _model_config(**overrides):
    values = {
        "hf_config": None,
        "model": "",
        "get_num_layers": lambda _parallel_config: 43,
        "get_num_kv_heads": lambda _parallel_config: 1,
        "get_head_size": lambda: 512,
        "get_max_model_len": lambda: 8192,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_detects_deepseek_v4_flash_from_hf_config() -> None:
    model_config = _model_config(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4FlashForCausalLM"],
        ),
    )

    adapter = get_model_adapter(object(), model_config)

    assert isinstance(adapter, DeepSeekV4FlashAdapter)
    assert adapter.model_type == "deepseek_v4_flash"


def test_registry_detects_deepseek_v4_flash_from_gguf_metadata(tmp_path) -> None:
    path = tmp_path / (
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    )
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )

    adapter = get_model_adapter(None, _model_config(model=str(path)))

    assert isinstance(adapter, DeepSeekV4FlashAdapter)


def test_deepseek_adapter_policy_is_experimental_and_capped() -> None:
    model_config = _model_config(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4FlashForCausalLM"],
        ),
    )
    adapter = get_model_adapter(object(), model_config)
    caps = adapter.detect(object(), model_config)
    policy = adapter.runtime_policy(model_config, SimpleNamespace())

    assert caps.supports_moe is True
    assert caps.supports_paged_prefill is False
    assert caps.supports_chunked_prefill is False
    assert caps.preferred_kv_dtype == "deepseek_v4_compressed"
    assert policy.model_policy["experimental"] is True
    assert policy.model_policy["max_tested_context"] == 8192
    assert policy.kernel_policy["compressed_attention_uses_page_tables"] is True


def test_deepseek_adapter_builds_custom_runtime_components() -> None:
    adapter = DeepSeekV4FlashAdapter()
    model = SimpleNamespace(
        raw_block_size=lambda: 16,
        num_raw_blocks_per_seq=lambda: 512,
        num_layers=lambda: 43,
    )

    components = adapter.build_executors(
        model=model,
        model_config=_model_config(),
        runtime_config=SimpleNamespace(kv_max_model_len=8192, kv_max_active_requests=2),
        observer=None,
        device="cpu",
        max_active_requests=2,
    )

    assert isinstance(components, CustomRuntimeComponents)
    assert isinstance(components.prefill_executor, DeepSeekPrefillExecutor)
    assert isinstance(components.decode_executor, DeepSeekDecodeExecutor)
    assert isinstance(components.kv_block_manager, DeepSeekKVLifecycleAdapter)
    assert components.multimodal_processor is None


def test_model_capabilities_default_to_chunked_prefill() -> None:
    from vllm.adapters.base import ModelCapabilities

    caps = ModelCapabilities(
        model_type="dummy",
        num_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=8,
        max_model_len=128,
        supports_moe=False,
        supports_fp8_kv=True,
        supports_int4_kv=True,
        supports_paged_prefill=True,
        preferred_kv_dtype="float16",
    )

    assert caps.supports_chunked_prefill is True
