# Troubleshooting (FastInference)

This document outlines troubleshooting strategies for FastInference (vLLM Lite). Since we have removed distributed complexity and C++ dependencies, most issues will be related to **Triton kernel compilation**, **GPU memory**, or **Model Loading**.

## Triton Kernel Issues

### Hangs during "Warmup"
FastInference performs a kernel warmup phase to JIT-compile Triton kernels.
- **Cause**: On some systems, the first compilation can take 30-60 seconds.
- **Solution**: Wait for the process to complete. Subsequent runs will use the cached kernels in `~/.triton/cache`.

### GPU Illegal Memory Access (Error 700)
If you see `hipErrorIllegalAddress` or `CUDA error: an illegal memory access was encountered`:
- **Cause**: This can happen on **AMD APUs** (like Strix Point) during high-concurrency Prefill.
- **Solution**: FastInference v2.0 includes a stability guard that automatically falls back to optimized PyTorch paths if Triton memory pressure is too high. Ensure you are using the latest `paged_attention.py` kernel.

## System Crashes

### Segfault (Signal 139) on AMD
If the process crashes immediately during `torch.zeros` allocation or model loading:
- **Cause**: Incompatibility between older ROCm versions and newer PyTorch memory allocators on `gfx1151`.
- **Solution**: **Upgrade to ROCm 7.2**. FastInference has been fully validated on ROCm 7.2, which resolves these low-level allocation crashes.

## Out of Memory (OOM)

### Weights loading OOM
- **Cause**: The model is too large for your GPU VRAM.
- **Solution**: Use GGUF or AWQ quantization. FastInference is highly optimized for **Self-Healing Loading**, which dequantizes weights block-by-block to minimize peak memory.

### KV Cache OOM
- **Cause**: Too many concurrent requests or very long context.
- **Solution**: Reduce `max_num_seqs` or `max_num_batched_tokens` in your engine configuration. You can also enable **FP8 KV Cache** to halve the memory usage.

## Model Loading Errors

### "RuntimeError: size mismatch"
- **Cause**: Loading a hybrid model (like Qwen3.5) with a standard Llama implementation.
- **Solution**: FastInference automatically routes Qwen3.5 to the `Qwen3_5LinearAttentionLayer`. If you manually created the model, ensure you are using the specialized backbone.

### "TypeError: not a string" during Tokenization
- **Cause**: AutoTokenizer failing to find GGUF-internal tokenizer files.
- **Solution**: FastInference includes a `Dummy` tokenizer fallback. To fix properly, ensure `tokenizer.json` is present in the model directory.

## Quality expectations (观感 vs strict HF alignment)

For a practical bar (“outputs look reasonable” vs bit-exact alignment with Hugging Face), see **[INFERENCE_ACCURACY.md](../INFERENCE_ACCURACY.md)** — recommended spot-check prompts and when lowering the strictness is **not** enough.

To run the tier-B spot-check script (Lite-only, fixed prompts): `uv run python scripts/quality_bar_spotcheck.py --model <path> --quant awq|gguf|none` (see INFERENCE_ACCURACY.md).

## Semantic integrity (`tests/verify_semantic_integrity.py`)

### Comparing Lite vs HF with the **same** checkpoint (`--hf-same-as-lite`)

Use this when you want to isolate **implementation** differences (Lite vs Hugging Face) instead of mixing in a separate FP16/BF16 tree:

```bash
uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-AWQ \
  --preset qwen35_9b_awq \
  --hf-same-as-lite
```

- If set, **`--hf-model` is ignored** and the HF reference is loaded from the same path as `--model`.
- **AWQ / packed checkpoints**: `transformers` may print a load report with **MISSING** / **UNEXPECTED** keys (e.g. `weight_packed` vs `weight`). In that case HF is **not** applying the same tensors as Lite, and prefill CosSim vs HF is **not meaningful** until HF uses a loader that matches the checkpoint format (e.g. the same compressed-tensors / AWQ path the hub expects).

### Comparing Lite vs an unquantized baseline

Use a separate FP16/BF16 directory:

```bash
uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-AWQ \
  --preset qwen35_9b_awq \
  --hf-model models/Qwen3.5-9B-FP16
```

Interpretation: low CosSim here can be **quantization + dtype** vs FP16, not only Lite bugs.

## Enabling Debug Logs

To see exactly what the Triton kernels and scheduler are doing:
- `export VLLM_LOGGING_LEVEL=DEBUG`
- `export AMD_SERIALIZE_KERNEL=3` (For AMD GPUs to catch the exact failing kernel)
