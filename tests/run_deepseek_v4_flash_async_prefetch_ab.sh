#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cat <<'EOF'
================================================================================
NOTE: This A/B script is kept as a historical record only.

The async expert prefetch runtime code was removed from the product branch
because the measured coverage (~4.8%) and end-to-end TPS improvement were too
small to justify the extra stream/event complexity.

The FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH environment variable is no
longer recognized by the runtime.

To reproduce the original experiment, check out the commit that introduced the
async prefetch feature (see docs/superpowers/specs/2026-07-07-deepseek-async-
prefetch-design.md).
================================================================================
EOF
exit 0
