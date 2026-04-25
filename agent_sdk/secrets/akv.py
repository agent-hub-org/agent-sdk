"""
Azure Key Vault bootstrap loader.

Call load_akv_secrets() at module level (before any other imports that read
os.environ). Fetches every enabled AKV secret and sets it as an env var
only if the var is not already set — so Docker env_file values take precedence.

Local dev:   az login, then set AZURE_KEY_VAULT_URL
Production:  set AZURE_CLIENT_ID + AZURE_TENANT_ID + AZURE_CLIENT_SECRET
             (or use managed identity — no credentials needed)
"""

import logging
import os

logger = logging.getLogger("agent_sdk.secrets.akv")


def load_akv_secrets() -> None:
    vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    if not vault_url:
        logger.debug("AZURE_KEY_VAULT_URL not set — skipping AKV bootstrap")
        return

    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
    except ImportError:
        logger.warning(
            "azure-keyvault-secrets / azure-identity not installed. "
            "Run: pip install 'agent-sdk[akv]'"
        )
        return

    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        loaded = 0
        for prop in client.list_properties_of_secrets():
            if not prop.enabled:
                continue
            env_key = prop.name.replace("-", "_").upper()
            if os.environ.get(env_key):
                continue  # don't overwrite existing values
            os.environ[env_key] = client.get_secret(prop.name).value
            loaded += 1
        logger.info("AKV bootstrap: loaded %d secret(s) from %s", loaded, vault_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AKV bootstrap failed (%s: %s) — continuing with existing env vars",
            type(exc).__name__, exc,
        )
