"""
Sentry initialization for agent-hub services.

Call init_sentry(service_name) in lifespan startup, after load_akv_secrets()
so SENTRY_DSN is available. No-op if SENTRY_DSN is not set.

Environment variables:
    SENTRY_DSN                 — Required to enable Sentry
    SENTRY_ENVIRONMENT         — Default: production
    SENTRY_TRACES_SAMPLE_RATE  — Float 0.0–1.0, default 0.1
    SERVICE_VERSION            — Release tag, default "unknown"
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("agent_sdk.observability.sentry")
_initialized = False


def init_sentry(service_name: str, release: Optional[str] = None) -> None:
    global _initialized
    if _initialized:
        return
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        logger.info("SENTRY_DSN not set — Sentry disabled for '%s'", service_name)
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        import logging as _logging

        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            release=release or os.getenv("SERVICE_VERSION", "unknown"),
            server_name=service_name,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            integrations=[
                StarletteIntegration(transaction_style="url"),
                FastApiIntegration(transaction_style="url"),
                LoggingIntegration(level=_logging.WARNING, event_level=_logging.ERROR),
            ],
            send_default_pii=False,
        )
        _initialized = True
        logger.info(
            "Sentry enabled for '%s' (env=%s)",
            service_name,
            os.getenv("SENTRY_ENVIRONMENT", "production"),
        )
    except ImportError:
        logger.warning("sentry-sdk not installed. Run: pip install 'agent-sdk[sentry]'")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentry init failed (%s) — continuing without error tracking", exc)


def set_request_user(user_id: Optional[str]) -> None:
    """Attach user_id to the current Sentry scope. Safe to call if Sentry is disabled."""
    if not _initialized or not user_id:
        return
    try:
        import sentry_sdk
        sentry_sdk.set_user({"id": user_id})
    except Exception:  # noqa: BLE001
        pass
