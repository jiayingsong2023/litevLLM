# AWQ Policy Matrix (9B / 35B)

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
| `qwen35_35b_awq` | `off` | `off` | `attention_only` | `off` |
| generic AWQ | `attention_only` | `all` | `all` | `attention_only` |

For Qwen3.5 35B AWQ high-fidelity forcing (`force_high_fidelity_awq`):

- `strict` -> `all`
- `balanced` -> `balanced`
- `throughput` -> `relaxed`
- `safe` -> `balanced`

This mapping is used only when `FASTINFERENCE_QWEN35_35B_AWQ_HIGH_FIDELITY_MODE` is not explicitly set.
