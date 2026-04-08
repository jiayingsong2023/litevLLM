# SPDX-License-Identifier: Apache-2.0


class LiteRuntimeError(RuntimeError):
    """Base runtime error for the lite engine."""


class RequestRejectedError(LiteRuntimeError):
    """Raised when a new request cannot be accepted."""


class RequestAbortedError(LiteRuntimeError):
    """Raised or propagated when a request is aborted."""


class BackgroundLoopError(LiteRuntimeError):
    """Raised or propagated when the async background loop fails."""


class ExecutionStepError(LiteRuntimeError):
    """Raised when a single prefill/decode execution step fails."""
