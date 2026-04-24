"""CPU vs GPU backend resolution (no model load)."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

os.environ.setdefault("CDTSM_AUTH_TOKEN", "test-key-default")

from app.core.config import Settings  # noqa: E402
from app.services.cisco_adapter import resolve_inference_backend  # noqa: E402

_DUMMY_AUTH = {"auth_token": "test-key-default"}


def test_resolve_cpu_forced():
    s = Settings(torch_backend="cpu", **_DUMMY_AUTH)
    assert resolve_inference_backend(s) == "cpu"


def test_resolve_auto_uses_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr("app.services.cisco_adapter.torch.cuda.is_available", lambda: False)
    s = Settings(torch_backend="auto", **_DUMMY_AUTH)
    assert resolve_inference_backend(s) == "cpu"


def test_resolve_auto_uses_gpu_when_cuda(monkeypatch):
    monkeypatch.setattr("app.services.cisco_adapter.torch.cuda.is_available", lambda: True)
    s = Settings(torch_backend="auto", **_DUMMY_AUTH)
    assert resolve_inference_backend(s) == "gpu"


def test_resolve_gpu_raises_without_cuda(monkeypatch):
    monkeypatch.setattr("app.services.cisco_adapter.torch.cuda.is_available", lambda: False)
    s = Settings(torch_backend="gpu", **_DUMMY_AUTH)
    with pytest.raises(RuntimeError) as e:
        resolve_inference_backend(s)
    assert "CUDA is not available" in str(e.value)


def test_config_rejects_invalid_torch_backend():
    with pytest.raises(ValidationError):
        Settings(torch_backend="cuda", **_DUMMY_AUTH)


def test_forecast_supports_restrict_quantiles_detects_current_package():
    """Probe should match whatever the currently installed cisco-tsm exposes."""
    import inspect as _inspect

    from cisco_tsm import CiscoTsmMR

    from app.services.cisco_adapter import forecast_supports_restrict_quantiles

    forecast_supports_restrict_quantiles.cache_clear()
    expected = "restrict_quantiles" in _inspect.signature(CiscoTsmMR.forecast).parameters
    assert forecast_supports_restrict_quantiles() is expected


def test_model_service_passes_restrict_quantiles_when_supported(monkeypatch):
    """ModelService.infer() adds restrict_quantiles=True iff the probe is True."""
    from unittest.mock import MagicMock

    import numpy as np

    from app.models.schemas import InferMetadata, InferRequestBody, SeriesPayload
    from app.services import model_service as ms

    monkeypatch.setattr(ms, "forecast_supports_restrict_quantiles", lambda: True)

    svc = ms.ModelService()
    svc.configure_from_settings(Settings(torch_backend="cpu", **_DUMMY_AUTH))
    mock_model = MagicMock()
    mock_model.forecast.return_value = [
        {"mean": np.array([1.0]), "quantiles": {"0.5": np.array([1.0])}}
    ]
    svc._model = mock_model
    svc._load_error = None

    body = InferRequestBody(
        payload=[SeriesPayload(coarse_ctx=[1.0], fine_ctx=[2.0])],
        metadata=InferMetadata(quantiles=["mean", "p50"]),
    )
    svc.infer(body, horizon=1)

    _, kwargs = mock_model.forecast.call_args
    assert kwargs.get("restrict_quantiles") is True


def test_model_service_omits_restrict_quantiles_when_unsupported(monkeypatch):
    """When the probe is False, the kwarg must not be forwarded."""
    from unittest.mock import MagicMock

    import numpy as np

    from app.models.schemas import InferMetadata, InferRequestBody, SeriesPayload
    from app.services import model_service as ms

    monkeypatch.setattr(ms, "forecast_supports_restrict_quantiles", lambda: False)

    svc = ms.ModelService()
    svc.configure_from_settings(Settings(torch_backend="cpu", **_DUMMY_AUTH))
    mock_model = MagicMock()
    mock_model.forecast.return_value = [
        {"mean": np.array([1.0]), "quantiles": {"0.5": np.array([1.0])}}
    ]
    svc._model = mock_model
    svc._load_error = None

    body = InferRequestBody(
        payload=[SeriesPayload(coarse_ctx=[1.0], fine_ctx=[2.0])],
        metadata=InferMetadata(quantiles=["mean", "p50"]),
    )
    svc.infer(body, horizon=1)

    _, kwargs = mock_model.forecast.call_args
    assert "restrict_quantiles" not in kwargs
