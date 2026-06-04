from types import SimpleNamespace

from vllm.model_executor.model_loader import _should_skip_safetensors_load


def test_deepseek_v4_flash_skips_safetensors_loader() -> None:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4FlashForCausalLM"],
            )
        )
    )

    assert _should_skip_safetensors_load(cfg) is True


def test_llama_does_not_skip_safetensors_loader() -> None:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            )
        )
    )

    assert _should_skip_safetensors_load(cfg) is False
