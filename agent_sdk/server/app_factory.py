"""FastAPI app factory — shared CORS / middleware setup for all agents."""
import os
from collections.abc import AsyncContextManager, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent_sdk.middleware.infra import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    VerifyInternalKeyMiddleware,
)
from agent_sdk.server.error_handlers import register_error_handlers


def create_agent_app(
    title: str,
    lifespan: AsyncContextManager,
    key_func: Callable | None = None,
) -> tuple[FastAPI, Limiter]:
    """Create a FastAPI app with all standard agent middleware pre-configured.

    Returns (app, limiter) where limiter is used for @limiter.limit() route decorators.

    Args:
        title: FastAPI app title.
        lifespan: Async context manager for startup/shutdown.
        key_func: Rate limiter key function (defaults to remote address).

    Wires:
    - slowapi rate limiter
    - CORS (driven by ALLOWED_ORIGINS env var)
    - SecurityHeadersMiddleware, RequestIDMiddleware, VerifyInternalKeyMiddleware
    - Standard error handlers
    """
    limiter = Limiter(key_func=key_func or get_remote_address)

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    register_error_handlers(app)

    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Internal-API-Key", "X-User-Id", "X-Request-ID"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(VerifyInternalKeyMiddleware)

    return app, limiter
