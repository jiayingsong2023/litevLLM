# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
def add_attention_backend(server_args, attention_config):
    """Append attention backend CLI arg if specified.

    Args:
        server_args: List of server arguments to extend in-place.
        attention_config: Dict with 'backend' key, or None.
    """
    if attention_config and "backend" in attention_config:
        server_args.extend(["--attention-backend", attention_config["backend"]])

