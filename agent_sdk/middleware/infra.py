import os
import uuid
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from agent_sdk.context import request_id_var, user_id_var

logger = logging.getLogger("agent_sdk.middleware.infra")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate X-Request-ID; set request_id_var and user_id_var for the request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        tok_r = request_id_var.set(request_id)
        tok_u = user_id_var.set(request.headers.get("X-User-Id"))
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(tok_r)
            user_id_var.reset(tok_u)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


_DEFAULT_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health", "/metrics", "/docs", "/openapi.json", "/a2a/.well-known/agent.json",
})


class VerifyInternalKeyMiddleware(BaseHTTPMiddleware):
    """Block non-public paths when INTERNAL_API_KEY env var is set and the header is wrong."""

    def __init__(self, app, public_paths: frozenset[str] = _DEFAULT_PUBLIC_PATHS):
        super().__init__(app)
        self.public_paths = public_paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in self.public_paths:
            expected = os.getenv("INTERNAL_API_KEY")
            if expected and request.headers.get("X-Internal-API-Key") != expected:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized internal access"},
                )
        return await call_next(request)
