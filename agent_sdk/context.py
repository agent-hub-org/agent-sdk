"""
Shared async context variables for request-scoped data propagation.

These ContextVars thread request_id and user_id through async call stacks
without explicit parameter passing, so log lines and downstream calls
(e.g. A2A delegations, MCP calls) can include correlation IDs automatically.

Usage in app.py middleware::

    from agent_sdk.context import request_id_var, user_id_var

    @app.middleware("http")
    async def inject_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tok_r = request_id_var.set(request_id)
        tok_u = user_id_var.set(request.headers.get("X-User-Id"))
        response = await call_next(request)
        request_id_var.reset(tok_r)
        user_id_var.reset(tok_u)
        response.headers["X-Request-ID"] = request_id
        return response
"""

from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
