# SPDX-License-Identifier: Apache-2.0
"""Text parsing utilities."""


def truthy(value: object) -> bool:
    """Return True if str(value) represents a truthy value."""
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")
