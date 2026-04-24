import os


def validate_required_env_vars(required: list[str], service_name: str) -> None:
    """Raise RuntimeError at startup if any required env vars are missing."""
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"[{service_name}] Missing required environment variables: {', '.join(missing)}\n"
            f"Set these before starting the service."
        )
