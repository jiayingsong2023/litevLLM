# Frequently Asked Questions (FastInference)

> Q: Why is this version called "Lite" or "FastInference"?

A: We have physically removed 70% of the original vLLM code (from 270k to 81k LOC). This includes removing all C++ extensions and distributed overhead to focus exclusively on single-GPU peak performance using OpenAI Triton.

---

> Q: Does FastInference require a C++ compiler?

A: **No.** Performance-critical kernels are written in Python using Triton. Use `uv sync` to install the pinned Python 3.12 environment; no project C++ build step is required.

---

> Q: How do I install and run FastInference locally?

A: Use Python 3.12 and run `uv sync` from the repository root. Project commands should use `uv run ...`; the maintained workflow does not use bare `python` or `pip`.

---

> Q: What should I do if I see "Illegal Memory Access" on AMD ROCm?

A: Reduce concurrency, prompt length, or KV limits first. The default large-model benchmark pins conservative single-request shapes for Gemma4 and enables per-model process isolation when benchmarking multiple large Gemma4 models in one command.

---

> Q: Can I run multiple LoRA adapters concurrently?

A: The LiteLoRA runtime exists and is treated as experimental. Validate your workload with smoke, correctness, and performance gates before treating a LoRA profile as production-ready.

---

> Q: Must inference match Hugging Face logits exactly?

A: **No** for many product use cases. You can aim for **semantically reasonable** outputs (readable, on-topic) instead of strict numerical alignment. See **[INFERENCE_ACCURACY.md](../INFERENCE_ACCURACY.md)** for two-tier expectations, spot-check prompts, and when garbled output still means a bug—not “lower expectations.”
