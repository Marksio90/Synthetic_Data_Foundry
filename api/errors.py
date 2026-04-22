from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    error_code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, request_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
            "request_id": request_id,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class ServiceUnavailableError(ApiError):
    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status_code=503, error_code=error_code, message=message, details=details or {})


class FailedDependencyError(ApiError):
    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status_code=424, error_code=error_code, message=message, details=details or {})
