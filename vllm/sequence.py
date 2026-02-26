# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own kv_connector_output.
