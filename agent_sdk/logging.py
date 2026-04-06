"""
Shared JSON structured logging for all agent services.

Usage in app.py (replaces the identical _JsonFormatter class copy-pasted in
every agent):

    from agent_sdk.logging import configure_logging
    configure_logging("agent_health")
"""

import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        # Inject request correlation IDs when set (avoids circular import by
        # importing lazily — context.py has no dependencies on logging.py)
        try:
            from agent_sdk.context import request_id_var, user_id_var  # noqa: PLC0415
            rid = request_id_var.get(None)
            uid = user_id_var.get(None)
            if rid:
                doc["request_id"] = rid
            if uid:
                doc["user_id"] = uid
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(doc, ensure_ascii=False)


def configure_logging(service_name: str | None = None) -> None:
    """
    Configure the root logger with JsonFormatter.

    Call once at application startup, before the FastAPI app is created.
    Reads LOG_LEVEL from the environment (default: INFO).

    Args:
        service_name: Optional name used in a startup log line to confirm
                      logging initialisation. Ignored if None.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # Avoid adding duplicate handlers if called multiple times
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter)
               for h in root.handlers):
        root.addHandler(handler)

    if service_name:
        logging.getLogger(service_name).info("Logging initialised for service '%s'", service_name)
