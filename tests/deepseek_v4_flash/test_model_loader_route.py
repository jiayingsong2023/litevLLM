from types import SimpleNamespace

import torch
from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.model_loader import (
    _should_skip_safetensors_load,
    get_model,
)
from vllm.model_executor.models.deepseek_v4_flash import DeepSeekV4FlashForCausalLM
from vllm.serving.config_builder import build_vllm_config


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


def test_get_model_constructs_deepseek_inspect_model_from_gguf(
    tmp_path,
    monkeypatch,
) -> None:
    filename = (
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    )
    path = tmp_path / filename
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: (_ for _ in ()).throw(
            AssertionError("DeepSeek inspect loader must not touch CUDA")
        ),
    )

    model = get_model(build_vllm_config(str(path), max_model_len=4096))
    try:
        assert isinstance(model, DeepSeekV4FlashForCausalLM)
        assert model.weight_store is not None
        assert model.weight_store.diagnostics.tensor_count == 2
        assert model.runtime_budget is not None
        assert model.runtime_budget.context.context_length == 4096
    finally:
        model.close()
