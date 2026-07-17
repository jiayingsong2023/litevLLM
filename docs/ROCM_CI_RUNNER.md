# ROCm CI runner contract

The static-quality job runs on GitHub-hosted Ubuntu and must not import Torch.
Inference, correctness, performance, and wheel-import smoke run only on a
self-hosted runner labelled `self-hosted`, `linux`, and `rocm`.

Before enabling the runner, install Python 3.12, `uv`, ROCm 7.2, and the
project's ROCm Torch/Triton wheels. Confirm this command succeeds:

```bash
uv sync --group dev
uv run python -c "import torch; assert torch.version.hip"
```

Set repository variable `FASTINFERENCE_ROCM_RUNNER=enabled` only after that
check and the model paths below are present on the runner:

- `models/gemma-4-26B-A4B-it-AWQ-4bit`
- `models/gemma-4-31B-it-AWQ-4bit`
- `models/DeepSeek-V4-Flash-ds4/...gguf`

Store the calibrated E2E baseline at
`/opt/fastinference/baselines/e2e-default.json`. Generate it on that runner
with the maintained isolated benchmark command, then review and replace it
only with an explicit performance-baseline PR.

After the first successful run, make `static-quality` and
`rocm-fast-regression` required checks for `main`. Add `rocm-performance` to
the PR only when it has the `rocm-performance` label; kernel, KV, scheduler,
or model hot-path changes must carry that label.
