from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from config.settings import settings


def require_admin_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    Enforce admin API key for sensitive endpoints.
    Fail closed when ADMIN_API_KEY is not configured.
    """
    expected = settings.admin_api_key.strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_API_KEY is not configured on the server.",
        )

    provided = (x_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )
