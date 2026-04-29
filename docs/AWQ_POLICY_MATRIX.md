# AWQ Policy Matrix (9B / Gemma4)

This project uses `FASTINFERENCE_AWQ_POLICY_MATRIX` to switch AWQ runtime defaults quickly without changing code.

## Quick Switch

```bash
# Balanced default
export FASTINFERENCE_AWQ_POLICY_MATRIX=balanced

# Throughput-biased
export FASTINFERENCE_AWQ_POLICY_MATRIX=throughput

# Strict fidelity
export FASTINFERENCE_AWQ_POLICY_MATRIX=strict

# Conservative safe mode
export FASTINFERENCE_AWQ_POLICY_MATRIX=safe
```

## Effective Defaults

`FASTINFERENCE_AWQ_FUSED_SCOPE` can still override the matrix directly.  
If `FASTINFERENCE_AWQ_FUSED_SCOPE` is unset, profile-aware defaults are:

| Model Profile | safe | balanced | throughput | strict |
|---|---|---|---|---|
| `qwen35_9b_awq` | `attention_only` | `attention_only` | `all` | `off` |
| `gemma4_31b_q4` | `attention_only` | `all` | `all` | `attention_only` |
| `gemma4_26b_a4b` | `attention_only` | `all` | `all` | `attention_only` |
| generic AWQ | `attention_only` | `all` | `all` | `attention_only` |

Gemma4 fused AWQ rollout is gated by `FASTINFERENCE_GEMMA4_FUSED_STAGE` (`off` / `attention_only` / `all`), which takes precedence over the policy matrix for Gemma4 models.
