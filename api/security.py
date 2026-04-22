from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, WebSocket, status

from config.settings import settings


def _verify_admin_api_key(provided_raw: str | None) -> bool:
    expected = settings.admin_api_key.strip()
    provided = (provided_raw or "").strip()
    return bool(expected and provided and hmac.compare_digest(provided, expected))


def require_admin_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    Enforce admin API key for sensitive endpoints.
    Fail closed when ADMIN_API_KEY is not configured.
    """
    if not settings.admin_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_API_KEY is not configured on the server.",
        )
    if not _verify_admin_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )


def require_admin_api_key_ws(websocket: WebSocket) -> bool:
    """
    WebSocket variant: accept API key from header (X-API-Key) or `api_key` query param.
    """
    if not settings.admin_api_key.strip():
        return False
    provided = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return _verify_admin_api_key(provided)
