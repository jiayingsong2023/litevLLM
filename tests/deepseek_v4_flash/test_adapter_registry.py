from types import SimpleNamespace

from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.adapters.deepseek_v4_flash import DeepSeekV4FlashAdapter
from vllm.adapters.registry import get_model_adapter


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
    assert caps.preferred_kv_dtype == "deepseek_v4_compressed"
    assert policy.model_policy["experimental"] is True
    assert policy.model_policy["max_tested_context"] == 8192
    assert policy.kernel_policy["compressed_attention_uses_page_tables"] is True
