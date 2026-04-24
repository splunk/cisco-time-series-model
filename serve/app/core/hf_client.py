"""Configure Hugging Face Hub HTTP client TLS settings."""

from __future__ import annotations

import logging

import httpx
import huggingface_hub.constants as hf_constants
from huggingface_hub.utils._http import (
    hf_request_event_hook,
    set_client_factory,
)

try:
    import truststore as _truststore
except ImportError:  # pragma: no cover - optional fallback on unusual platforms
    _truststore = None

from app.core.config import Settings

logger = logging.getLogger(__name__)


def configure_huggingface_http(settings: Settings) -> None:
    """Set up TLS verification and timeouts for huggingface_hub HTTP calls."""
    if _truststore is not None:
        _truststore.inject_into_ssl()
    else:
        logger.info("truststore not installed; using certifi CA bundle for TLS verification")

    merged = max(settings.hf_hub_download_timeout_seconds, hf_constants.HF_HUB_DOWNLOAD_TIMEOUT)
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = merged
    logger.info("Hugging Face HF_HUB_DOWNLOAD_TIMEOUT=%ss", merged)

    if settings.hf_insecure_ssl:
        verify: bool | str = False
        logger.warning(
            "HF TLS verification is disabled (CDTSM_HF_INSECURE_SSL); use only in trusted networks"
        )
    elif settings.hf_ssl_ca_bundle:
        verify = settings.hf_ssl_ca_bundle
    else:
        verify = True

    def factory() -> httpx.Client:
        return httpx.Client(
            verify=verify,
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    set_client_factory(factory)
