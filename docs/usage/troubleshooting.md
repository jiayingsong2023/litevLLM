# Troubleshooting (FastInference)

This document outlines troubleshooting strategies for FastInference (vLLM Lite). Since we have removed distributed complexity and C++ dependencies, most issues will be related to **Triton kernel compilation**, **GPU memory**, or **Model Loading**.

## Triton Kernel Issues

### Hangs during "Warmup"
FastInference performs a kernel warmup phase to JIT-compile Triton kernels.
- **Cause**: On some systems, the first compilation can take 30-60 seconds.
- **Solution**: Wait for the process to complete. Subsequent runs will use the cached kernels in `~/.triton/cache`.

### GPU Illegal Memory Access (Error 700)
If you see `hipErrorIllegalAddress` or `CUDA error: an illegal memory access was encountered`:
- **Cause**: This often happens on **AMD APUs** (like Strix Point) when the Batch Size is too high for the memory controller to handle fragmented Paged KV Cache writes.
- **Solution**: We have implemented a stability patch using PyTorch Advanced Indexing. If you still encounter this, try reducing the Batch Size to 8 or 16.

## Out of Memory (OOM)

### Weights loading OOM
- **Cause**: The model is too large for your GPU VRAM.
- **Solution**: Use GGUF quantization. FastInference is highly optimized for GGUF and uses an **LRU Weight Cache** to keep only active layers in VRAM.

### KV Cache OOM
- **Cause**: Too many concurrent requests or very long context.
- **Solution**: Reduce `max_num_seqs` or `max_num_batched_tokens` in your engine configuration. You can also enable **FP8 KV Cache** to halve the memory usage.

## Model Loading Errors

### "AttributeError: 'LiteLinear' object has no attribute 'qweight'"
- **Cause**: Attempting to load a quantized model without a proper `quant_config`.
- **Solution**: Ensure your model uses a supported format (GGUF, AWQ, FP8). Check if the model directory contains the required quantization metadata.

### Missing Tokenizer
- **Cause**: FastInference uses a global `TokenizerRegistry`. If the model path is incorrect, it will fail to load.
- **Solution**: Ensure the `--model` path points to a valid HuggingFace-style directory containing `tokenizer.json`.

## Enabling Debug Logs

To see exactly what the Triton kernels and scheduler are doing:
- `export VLLM_LOGGING_LEVEL=DEBUG`
- `export AMD_SERIALIZE_KERNEL=3` (For AMD GPUs to catch the exact failing kernel)
