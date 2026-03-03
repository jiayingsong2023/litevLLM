# Frequently Asked Questions (FastInference)

> Q: Why is this version called "Lite" or "FastInference"?

A: We have physically removed 70% of the original vLLM code (from 270k to 81k LOC). This includes removing all C++ extensions and distributed overhead to focus exclusively on single-GPU peak performance using OpenAI Triton.

---

> Q: Does FastInference require a C++ compiler?

A: **No.** This is a major advantage. All performance-critical kernels (Attention, RMSNorm, RoPE, Dequantization) are written in Python using Triton. You can install and run the engine using only `uv pip install -e .`.

---

> Q: How does GGUF performance compare to standard vLLM?

A: FastInference is significantly faster for GGUF on single GPUs. By using a **Global LRU Cache** for dequantized weights, we avoid the overhead of repeated dequantization. Llama-7B GGUF reaches **195+ TPS** on an AMD Strix Point APU.

---

> Q: What should I do if I see "Illegal Memory Access" on my AMD APU?

A: This is a known hardware constraint when running large batches. We have included a stability patch. If the error persists, reduce your Batch Size to 8 or 16. FastInference still provides over **140 TPS** at Batch Size 8 for MoE models.

---

> Q: Can I run multiple LoRA adapters concurrently?

A: Yes. Our **Multi-adapter LiteLoRA** architecture supports routing different tokens in the same batch to different adapters with zero-copy overhead. We have benchmarked this at **663 TPS** with 3 concurrent adapters.
