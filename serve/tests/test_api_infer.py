from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Force the auth token used for tests to a known value. ``os.environ[...] = ...``
# (not ``setdefault``) guarantees we override any value the caller may have
# exported (e.g. a real CDTSM_AUTH_TOKEN from ``make model-up``) or that
# ``.env`` might otherwise inject via pydantic-settings.
os.environ["CDTSM_AUTH_TOKEN"] = "test-key-default"

from app.core.config import get_settings  # noqa: E402
from app.core.quantiles import resolve_requested_quantiles  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import InferRequestBody, PredictionItem  # noqa: E402
from app.services.model_service import get_model_service  # noqa: E402


class _ReadyMockService:
    def __init__(self) -> None:
        self._load_error = None

    def load(self) -> None:
        pass

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def load_error(self) -> None:
        return None

    @property
    def inference_backend(self) -> str:
        return "cpu"

    def model_load_status_dict(self) -> dict:
        return {
            "phase": "succeeded",
            "started_at": "2020-01-01T00:00:00+00:00",
            "completed_at": "2020-01-01T00:00:01+00:00",
            "elapsed_seconds": 1.0,
        }

    def infer(self, body: InferRequestBody, horizon: int) -> list[PredictionItem]:
        resolve_requested_quantiles(body.metadata.quantiles)
        out: list[PredictionItem] = []
        for _ in body.payload:
            mean = np.linspace(0, 1, horizon, dtype=np.float32).tolist()
            q40 = np.linspace(0, 0.5, horizon, dtype=np.float32).tolist()
            q90 = np.linspace(0, 0.9, horizon, dtype=np.float32).tolist()
            qnames = [q for q in body.metadata.quantiles if q != "mean"]
            quantiles: dict[str, list[float]] = {}
            if "p40" in qnames:
                quantiles["p40"] = q40
            if "p90" in qnames:
                quantiles["p90"] = q90
            out.append(PredictionItem(mean=mean, quantiles=quantiles))
        return out


_AUTH_HEADER = {"Authorization": "Bearer test-key-default"}


@pytest.fixture()
def client(monkeypatch):
    mock = _ReadyMockService()
    app.dependency_overrides[get_model_service] = lambda: mock
    monkeypatch.setattr("app.main.get_model_service", lambda: mock)
    with TestClient(app, headers=_AUTH_HEADER) as c:
        try:
            yield c
        finally:
            app.dependency_overrides.pop(get_model_service, None)


def test_infer_multi_series_payload(client: TestClient):
    """POST /cdtsm/v1/ai/infer with a multi-series batch (mirrors the README curl example)."""
    payload = {
        "payload": [
            {
                "coarse_ctx": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "fine_ctx": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            },
            {
                "coarse_ctx": [10.0, 20.0, 30.0, 40.0],
                "fine_ctx": [15.0, 25.0, 35.0, 45.0],
            },
        ],
        "model": "CDTSM",
        "metadata": {"quantiles": ["mean", "p40", "p90"]},
    }
    horizon = 12
    r = client.post(
        f"/cdtsm/v1/ai/infer?horizon={horizon}",
        json=payload,
        headers={"request_id": "multi-series-1", "Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["request_id"] == "multi-series-1"
    assert data["model"] == "CDTSM"
    assert data["horizon"] == horizon
    assert len(data["predictions"]) == len(payload["payload"])
    for pred in data["predictions"]:
        assert len(pred["mean"]) == horizon
        assert "p40" in pred["quantiles"]
        assert "p90" in pred["quantiles"]


def test_infer_success(client: TestClient):
    payload = {
        "payload": [
            {"coarse_ctx": [1.0, 2.0], "fine_ctx": [3.0, 4.0]},
        ],
        "model": "CDTSM",
        "metadata": {"quantiles": ["mean", "p40", "p90"]},
    }
    r = client.post(
        "/cdtsm/v1/ai/infer?horizon=8",
        json=payload,
        headers={"request_id": "123", "Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["request_id"] == "123"
    assert data["model"] == "CDTSM"
    assert data["horizon"] == 8
    assert len(data["predictions"]) == 1
    assert len(data["predictions"][0]["mean"]) == 8
    assert "p40" in data["predictions"][0]["quantiles"]


def test_infer_accepts_json_null_samples(client: TestClient):
    """JSON ``null`` in a context series round-trips through validation; the
    upstream model handles missing samples via interpolation."""
    payload = {
        "payload": [
            {"coarse_ctx": [1.0, None, 3.0], "fine_ctx": [1.5, 2.5, 3.5]},
        ],
        "model": "CDTSM",
        "metadata": {"quantiles": ["mean"]},
    }
    r = client.post(
        "/cdtsm/v1/ai/infer?horizon=4",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["predictions"][0]["mean"]) == 4


def test_infer_auth_token_rejected(monkeypatch):
    monkeypatch.setenv("CDTSM_AUTH_TOKEN", "test-key-123")
    get_settings.cache_clear()
    mock = _ReadyMockService()
    app.dependency_overrides[get_model_service] = lambda: mock
    monkeypatch.setattr("app.main.get_model_service", lambda: mock)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/cdtsm/v1/ai/infer?horizon=4",
                json={
                    "payload": [{"coarse_ctx": [1.0], "fine_ctx": [2.0]}],
                    "model": "CDTSM",
                    "metadata": {"quantiles": ["mean"]},
                },
            )
            assert r.status_code == 401

            r2 = client.post(
                "/cdtsm/v1/ai/infer?horizon=4",
                headers={"Authorization": "Bearer test-key-123"},
                json={
                    "payload": [{"coarse_ctx": [1.0], "fine_ctx": [2.0]}],
                    "model": "CDTSM",
                    "metadata": {"quantiles": ["mean"]},
                },
            )
            assert r2.status_code == 200
    finally:
        app.dependency_overrides.pop(get_model_service, None)
    get_settings.cache_clear()
    monkeypatch.delenv("CDTSM_AUTH_TOKEN", raising=False)


def test_infer_503_while_model_still_loading(monkeypatch):
    class _LoadingMockService:
        def load(self) -> None:
            pass

        @property
        def is_ready(self) -> bool:
            return False

        @property
        def load_error(self) -> None:
            return None

        def model_load_status_dict(self) -> dict:
            return {"phase": "in_progress", "started_at": "2020-01-01T00:00:00+00:00", "completed_at": None, "elapsed_seconds": 99.0}

        def infer(self, body, horizon):
            raise AssertionError("infer must not run while loading")

    loading = _LoadingMockService()
    app.dependency_overrides[get_model_service] = lambda: loading
    monkeypatch.setattr("app.main.get_model_service", lambda: loading)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/cdtsm/v1/ai/infer?horizon=4",
                json={
                    "payload": [{"coarse_ctx": [1.0], "fine_ctx": [2.0]}],
                    "model": "CDTSM",
                    "metadata": {"quantiles": ["mean"]},
                },
                headers={"request_id": "123", **_AUTH_HEADER},
            )
            assert r.status_code == 503, r.text
            body = r.json()
            assert body["error"]["code"] == "model_not_ready"
            assert "still loading" in body["error"]["message"].lower()
    finally:
        app.dependency_overrides.pop(get_model_service, None)


def test_ready_503_while_model_still_loading(monkeypatch):
    class _LoadingMockService:
        def load(self) -> None:
            pass

        @property
        def is_ready(self) -> bool:
            return False

        @property
        def load_error(self) -> None:
            return None

        def model_load_status_dict(self) -> dict:
            return {"phase": "in_progress", "started_at": "2020-01-01T00:00:00+00:00", "completed_at": None, "elapsed_seconds": 42.0}

    loading = _LoadingMockService()
    app.dependency_overrides[get_model_service] = lambda: loading
    monkeypatch.setattr("app.main.get_model_service", lambda: loading)
    try:
        with TestClient(app) as client:
            r = client.get("/ready")
            assert r.status_code == 503, r.text
            data = r.json()
            assert data["model_load"]["phase"] == "in_progress"
            assert data["model_load"]["elapsed_seconds"] == 42.0
            assert "loading" in data["message"].lower()
    finally:
        app.dependency_overrides.pop(get_model_service, None)


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["ready"] == "/ready"
    assert data["endpoints"]["docs"] == "/docs"


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_when_loaded(client: TestClient):
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data["inference_backend"] == "cpu"
    assert data["torch_backend"] in ("cpu", "gpu", "auto")
    assert data["model_load"]["phase"] == "succeeded"
    assert data["model_load"]["elapsed_seconds"] == 1.0


def test_infer_rejects_bad_quantile(client: TestClient):
    payload = {
        "payload": [{"coarse_ctx": [1.0], "fine_ctx": [2.0]}],
        "model": "CDTSM",
        "metadata": {"quantiles": ["p77"]},
    }
    r = client.post("/cdtsm/v1/ai/infer?horizon=4", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "bad_input"
