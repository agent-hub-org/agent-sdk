"""
Keycloak JWKS-based JWT validation middleware.

Drop-in replacement for the custom HS256 _JWTDecodeMiddleware.
Validates Keycloak RS256 access tokens via JWKS and sets
request.state.user_id = sub claim (Keycloak user UUID string).

JWKS is cached in-process for 5 minutes to avoid per-request HTTP calls.

Environment variables:
    KEYCLOAK_URL       — e.g. http://keycloak:8180 (no trailing slash)
    KEYCLOAK_REALM     — e.g. agent-hub
    KEYCLOAK_AUDIENCE  — e.g. agent-hub-frontend (empty = skip audience check)
"""

import logging
import os
import time
from typing import Optional

import httpx
from jose import jwt as _jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("agent_sdk.auth.keycloak")

_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 300.0  # seconds


async def _get_jwks(keycloak_url: str, realm: str) -> dict:
    global _jwks_cache, _jwks_fetched_at
    now = time.monotonic()
    if _jwks_cache and (now - _jwks_fetched_at) < _JWKS_TTL:
        return _jwks_cache
    url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_fetched_at = now
        logger.debug("JWKS refreshed from %s", url)
    return _jwks_cache


def _extract_user_id(token: str, jwks: dict, audience: Optional[str]) -> Optional[str]:
    try:
        payload = _jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=audience,
            options={"verify_aud": bool(audience)},
        )
        return payload.get("sub") or None
    except JWTError:
        return None


class KeycloakJWTMiddleware(BaseHTTPMiddleware):
    """
    Sets request.state.user_id from a Keycloak RS256 access token.
    Reads token from Authorization: Bearer header or access_token cookie.
    Sets user_id = None for unauthenticated requests; endpoints enforce auth themselves.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._url = os.getenv("KEYCLOAK_URL", "").rstrip("/")
        self._realm = os.getenv("KEYCLOAK_REALM", "agent-hub")
        self._audience = os.getenv("KEYCLOAK_AUDIENCE", "") or None
        if not self._url:
            logger.warning("KEYCLOAK_URL not set — all requests will be unauthenticated")

    async def dispatch(self, request: Request, call_next):
        request.state.user_id = None
        if not self._url:
            return await call_next(request)

        token = (
            request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or request.cookies.get("access_token")
            or None
        )

        if token:
            try:
                jwks = await _get_jwks(self._url, self._realm)
                request.state.user_id = _extract_user_id(token, jwks, self._audience)
            except Exception as exc:  # noqa: BLE001
                logger.warning("JWKS fetch failed (%s) — unauthenticated", exc)

        from agent_sdk.observability.sentry import set_request_user
        set_request_user(request.state.user_id)

        return await call_next(request)
