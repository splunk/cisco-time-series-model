from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.security import HTTPBearer

from app.core.config import Settings
from app.core.exceptions import AIPlatformAuthenticationError

# Documentation-only security scheme. ``auto_error=False`` keeps the real
# validation (and our standardized error envelope) in ``verify_auth_token``;
# this instance exists so Swagger UI / ReDoc render the "Authorize" button
# and the padlock icon on protected endpoints.
auth_scheme = HTTPBearer(
    bearerFormat="CDTSM_AUTH_TOKEN",
    description="Paste the value of CDTSM_AUTH_TOKEN (no 'Bearer ' prefix).",
    auto_error=False,
)


def verify_auth_token(settings: Settings, request: Request) -> None:
    auth = request.headers.get("authorization") or ""
    prefix = "Bearer "
    if not auth.startswith(prefix):
        raise AIPlatformAuthenticationError("Missing or invalid Authorization header")
    token = auth[len(prefix):].strip()
    if not token or not hmac.compare_digest(token, settings.auth_token):
        raise AIPlatformAuthenticationError("Invalid auth token")
