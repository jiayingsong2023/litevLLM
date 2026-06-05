import subprocess
import sys

from fixtures import write_minimal_deepseek_v4_flash_gguf


def test_inspect_tool_prints_shape_and_diagnostics(tmp_path) -> None:
    path = tmp_path / "model.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )

    result = subprocess.run(
        [sys.executable, "tests/tools/deepseek_v4_flash_inspect.py", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "DeepSeek V4 Flash" in result.stdout
    assert "layers: 43" in result.stdout
    assert "bound tensors: 2" in result.stdout
    assert "tensor types: 8=2" in result.stdout
    assert "unaligned tensor offsets: none" in result.stdout
    assert "model mmap bytes:" in result.stdout
    assert "resident bytes:" in result.stdout
    assert "available UMA headroom:" in result.stdout
    assert "context 8192:" in result.stdout
