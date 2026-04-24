"""Optional heavyweight test: real HF download + torch model load.

Skipped by default. Run with ``RUN_MODEL_LOAD=1 pytest`` to exercise end-to-end.
If TLS fails, set ``CDTSM_HF_INSECURE_SSL=1`` or ``CDTSM_HF_SSL_CA_BUNDLE``.
"""

from __future__ import annotations

import os

os.environ.setdefault("CDTSM_AUTH_TOKEN", "test-key-default")

import pytest  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.hf_client import configure_huggingface_http  # noqa: E402
from app.services.cisco_adapter import build_cisco_tsm_model  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_LOAD") != "1",
    reason="Set RUN_MODEL_LOAD=1 to run heavyweight model load test",
)


def test_build_cisco_tsm_model_smoke():
    s = Settings(
        hf_repo_id=os.environ.get("CDTSM_HF_REPO_ID", "cisco-ai/cisco-time-series-model-1.0"),
        torch_backend="cpu",
        auth_token=os.environ.get("CDTSM_AUTH_TOKEN", "test-key-default"),
    )
    configure_huggingface_http(s)
    model, backend = build_cisco_tsm_model(s)
    assert model is not None
    assert backend in ("cpu", "gpu")
