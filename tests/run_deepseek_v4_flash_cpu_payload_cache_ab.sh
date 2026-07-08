#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cat <<'EOF'
================================================================================
NOTE: This A/B script is kept as a historical record only.

The CPU payload cache runtime code was removed from the product branch
because it produced zero cache hits on the 96 GB UMA test machine (the default
GPU staging budget already held the entire working set) and added eviction
churn without end-to-end TPS improvement.

The FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES environment
variable is no longer recognized by the runtime.

To reproduce the original experiment, check out the commit that introduced the
CPU payload cache feature (see docs/performance_evaluation_deepseek_v4_flash_
2026_07_08.md).
================================================================================
EOF
exit 0
