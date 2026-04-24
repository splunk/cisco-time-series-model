from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.model_service import reset_model_service_for_tests


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    reset_model_service_for_tests()
    get_settings.cache_clear()
    yield
    reset_model_service_for_tests()
    get_settings.cache_clear()
