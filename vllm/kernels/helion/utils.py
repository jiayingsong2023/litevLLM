# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    Canonicalize GPU name for use as a platform identifier.

    Converts to lowercase and replaces spaces and hyphens with underscores.
    e.g., "NVIDIA A100-SXM4-80GB" -> "nvidia_a100_sxm4_80gb"

    Raises ValueError if name is empty.
