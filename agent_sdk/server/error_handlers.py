"""
FastAPI exception handlers for consistent AgentError envelope across all agents.

Usage in each agent's app.py:
    from agent_sdk.server.error_handlers import register_error_handlers
    register_error_handlers(app)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent_sdk.errors import AgentError, ErrorCode
from agent_sdk.context import request_id_var

logger = logging.getLogger("agent_sdk.error_handlers")


def register_error_handlers(app: FastAPI) -> None:
    """Register structured error handlers on a FastAPI app."""

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        exc.request_id = exc.request_id or request_id_var.get(None)
        logger.error("AgentError [%s]: %s (request_id=%s)", exc.error_code, exc.message, exc.request_id)
        return JSONResponse(status_code=500, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request_id_var.get(None)
        logger.error("Unhandled exception (request_id=%s): %s", request_id, exc, exc_info=True)
        try:
            from agent_sdk.observability.sentry import _initialized
            if _initialized:
                import sentry_sdk
                sentry_sdk.capture_exception(exc)
        except Exception:  # noqa: BLE001
            pass
        envelope = AgentError(
            error_code=ErrorCode.INTERNAL,
            message="An unexpected error occurred. Please try again.",
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=envelope.to_dict())
