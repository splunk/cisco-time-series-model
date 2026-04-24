"""Domain errors mapped to HTTP responses."""

from __future__ import annotations


class CDTSMServiceError(Exception):
    """Base error for API failures."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BadInputError(CDTSMServiceError):
    status_code = 400
    code = "bad_input"


class AIPlatformAuthenticationError(CDTSMServiceError):
    status_code = 401
    code = "authentication_failed"


class AIPlatformAuthorizationError(CDTSMServiceError):
    status_code = 403
    code = "authorization_failed"


class ModelNotFoundError(CDTSMServiceError):
    status_code = 404
    code = "model_not_found"


class ModelNotReadyError(CDTSMServiceError):
    status_code = 503
    code = "model_not_ready"


class AIPlatformRateLimitError(CDTSMServiceError):
    status_code = 429
    code = "rate_limited"


HTTP_STATUS_EXCEPTION_MAP: dict[int, type[CDTSMServiceError]] = {
    400: BadInputError,
    401: AIPlatformAuthenticationError,
    403: AIPlatformAuthorizationError,
    404: ModelNotFoundError,
    429: AIPlatformRateLimitError,
    503: ModelNotReadyError,
}
