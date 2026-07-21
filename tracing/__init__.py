# ------------------------------------------------------------------
# Public API imports (absolute, explicit, stable)
# ------------------------------------------------------------------

from tracing.api import (
    AgentSession,
    trace_step,
    update_step_metadata,
    emit_stream_chunk,
)

from tracing.infra.transport import (
    ConsoleJsonTransport,
    WebSocketTransport,
)

# ------------------------------------------------------------------
# Explicit public API surface
# ------------------------------------------------------------------

__all__ = [
    # Session lifecycle
    "AgentSession",

    # Business-level tracing
    "trace_step",

    # Decoupled data & streaming APIs
    "update_step_metadata",
    "emit_stream_chunk",

    # Transport abstractions
    "ConsoleJsonTransport",
    "WebSocketTransport",
]