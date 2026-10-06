"""Execution context for LLM logs (never sent to models)."""
from contextlib import contextmanager
from contextvars import ContextVar


_CONTEXT = ContextVar("llm_request_log_context", default={})


def current_request_context() -> dict:
    return dict(_CONTEXT.get())


@contextmanager
def request_log_context(**fields):
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)
