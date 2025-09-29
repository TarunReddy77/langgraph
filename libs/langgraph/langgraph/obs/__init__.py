"""Observability integrations for LangGraph."""

from __future__ import annotations

__all__ = [
    "LangfuseIngestionHandler",
    "make_langfuse_handler_from_env",
]

try:
    from .langfuse import LangfuseIngestionHandler, make_langfuse_handler_from_env
except Exception:  # pragma: no cover - avoid hard import failures at import time
    # Defer errors to runtime when/if the handler is constructed.
    LangfuseIngestionHandler = None  # type: ignore

    def make_langfuse_handler_from_env():  # type: ignore
        return None
