from __future__ import annotations

from contextlib import contextmanager
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_service


V1_INFER = "/cdtsm/v1/ai/infer"
V15_INFER = "/cdtsm/v1.5/ai/infer"
V15_READY = "/cdtsm/v1.5/ready"
AUTH_HEADER = {"Authorization": "Bearer v15-test-token"}
CLOCK = ["2025-01-06T00:00:00", "2025-01-06T00:05:00", "2025-01-06T00:10:00"]


def forecast_item(level, horizon):
    median = np.full(horizon, level, dtype=np.float32)
    return {"mean": median, "quantiles": {"0.5": median, "0.9": median + 2}}


@contextmanager
def configured_server(monkeypatch, tmp_path, *, enabled=True, failed=False):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CDTSM_AUTH_TOKEN", "v15-test-token")
    monkeypatch.setenv("CDTSM_TORCH_BACKEND", "cpu")
    monkeypatch.setenv("CDTSM_INFER_BATCH_SIZE", "2")
    if enabled:
        checkpoint = tmp_path / "ctsm-1.5-weights.pt"
        checkpoint.touch()
        monkeypatch.setenv("CDTSM_V15_CHECKPOINT_PATH", str(checkpoint))
    else:
        monkeypatch.delenv("CDTSM_V15_CHECKPOINT_PATH", raising=False)

    legacy = Mock()
    legacy.forecast.side_effect = lambda pairs, **kwargs: [
        forecast_item(fine[0], kwargs["horizon_len"]) for _, fine in pairs
    ]
    v15 = Mock()
    v15.forecast.side_effect = lambda histories, **kwargs: [
        forecast_item(history[0], kwargs["horizon_len"]) for history in histories
    ]
    v15_loader = Mock(return_value=(v15, "cpu"))
    if failed:
        v15_loader.side_effect = RuntimeError("unreadable CTSM 1.5 weights")
    monkeypatch.setattr(model_service, "build_cisco_tsm_model", lambda settings: (legacy, "cpu"))
    monkeypatch.setattr(model_service, "build_ctsm_v15_model", v15_loader)

    with TestClient(app, headers=AUTH_HEADER) as client:
        yield client, legacy, v15, v15_loader


def wait_for_load(client, route, phase):
    deadline = monotonic() + 5
    while True:
        response = client.get(route)
        if response.json().get("model_load", {}).get("phase") == phase:
            return response
        if monotonic() >= deadline:
            pytest.fail(f"{route} did not reach {phase}: {response.text}")
        sleep(0.01)


def test_legacy_route_and_readiness_work_without_v15(monkeypatch, tmp_path):
    with configured_server(monkeypatch, tmp_path, enabled=False) as (client, legacy, _, load_v15):
        assert wait_for_load(client, "/ready", "succeeded").status_code == 200
        response = client.post(
            V1_INFER + "?horizon=2",
            json={"payload": [{"coarse_ctx": [1, 2], "fine_ctx": [3, 4]}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "CDTSM"
        assert response.json()["predictions"] == [{"mean": [3.0, 3.0], "quantiles": {"p50": [3.0, 3.0]}}]
        assert legacy.forecast.call_args.args[0] == [([1.0, 2.0], [3.0, 4.0])]
        for response in (
            client.get(V15_READY),
            client.post(V15_INFER, json={"payload": [{"history": [1, 2]}]}),
        ):
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "model_not_found"
        load_v15.assert_not_called()


def test_v15_batches_clocked_and_value_only_series_in_request_order(monkeypatch, tmp_path):
    with configured_server(monkeypatch, tmp_path) as (client, legacy, v15, _):
        assert wait_for_load(client, V15_READY, "succeeded").status_code == 200
        response = client.post(
            V15_INFER + "?horizon=3",
            headers={"request_id": "mixed-v15"},
            json={
                "payload": [
                    {"history": [10, None, 12], "timestamps": CLOCK},
                    {"history": [20, 21]},
                    {"history": [30, 31, 32], "timestamps": CLOCK},
                ],
                "metadata": {"quantiles": ["p50", "p90"]},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["request_id"] == "mixed-v15"
        assert result["model"] == "CTSM-1.5"
        assert result["horizon"] == 3
        for level, prediction in zip((10, 20, 30), result["predictions"], strict=True):
            assert prediction == {
                "mean": [float(level)] * 3,
                "quantiles": {"p50": [float(level)] * 3, "p90": [float(level + 2)] * 3},
            }
        calls = {call.kwargs["timestamps"] is not None: call for call in v15.forecast.call_args_list}
        assert v15.forecast.call_count == 2
        assert all("use_timestamps" not in call.kwargs for call in calls.values())
        assert {mode: [history[0] for history in call.args[0]] for mode, call in calls.items()} == {
            True: [10, 30], False: [20],
        }
        assert all(call.kwargs["batch_size"] == 2 for call in calls.values())
        assert np.isnan(calls[True].args[0][0][1])
        assert len(calls[True].kwargs["timestamps"]) == 2
        legacy.forecast.assert_not_called()
        assert wait_for_load(client, "/ready", "succeeded").status_code == 200
        response = client.post(
            V1_INFER + "?horizon=1", json={"payload": [{"coarse_ctx": [1], "fine_ctx": [2]}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "CDTSM"
        assert v15.forecast.call_count == 2


@pytest.mark.parametrize(
    ("series", "status", "error_code"),
    [
        ({"history": []}, 422, "validation_error"),
        ({"history": [1, "invalid"]}, 422, "validation_error"),
        ({"history": [1, 2, 3], "timestamps": CLOCK[:2]}, 400, "bad_input"),
        ({"history": [1, 2, 3], "timestamps": CLOCK[::-1]}, 400, "bad_input"),
        ({"history": [1, 2, 3], "timestamps": [time + "Z" for time in CLOCK]}, 400, "bad_input"),
    ],
)
def test_v15_validates_history_and_timestamps(monkeypatch, tmp_path, series, status, error_code):
    with configured_server(monkeypatch, tmp_path) as (client, _, v15, _):
        wait_for_load(client, V15_READY, "succeeded")
        response = client.post(V15_INFER, json={"payload": [series]})
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == error_code
        v15.forecast.assert_not_called()


@pytest.mark.parametrize(
    ("route", "ready", "series"),
    [
        (V1_INFER, "/ready", {"coarse_ctx": [1], "fine_ctx": [1, 2]}),
        (V15_INFER, V15_READY, {"history": [1, 2]}),
    ],
)
@pytest.mark.parametrize(
    ("auth", "status"),
    [(b"bEaReR v15-test-token", 200), (b"Bearer invalid", 401), (b"Bearer \xe9", 401)],
)
def test_infer_endpoints_use_bearer_auth(monkeypatch, tmp_path, route, ready, series, auth, status):
    with configured_server(monkeypatch, tmp_path) as (client, legacy, v15, _):
        wait_for_load(client, ready, "succeeded")
        response = client.post(
            route,
            headers={b"Authorization": auth},
            json={"payload": [series]},
        )
        assert response.status_code == status, response.text
        if status == 401:
            assert response.json()["error"]["code"] == "authentication_failed"
            legacy.forecast.assert_not_called()
            v15.forecast.assert_not_called()


def test_v15_load_failure_does_not_block_legacy_inference(monkeypatch, tmp_path):
    with configured_server(monkeypatch, tmp_path, failed=True) as (client, _, _, _):
        assert wait_for_load(client, V15_READY, "failed").status_code == 503
        assert wait_for_load(client, "/ready", "succeeded").status_code == 200
        response = client.post(V15_INFER, json={"payload": [{"history": [1, 2]}]})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"
        assert client.post(
            V1_INFER + "?horizon=1", json={"payload": [{"coarse_ctx": [1], "fine_ctx": [2]}]},
        ).status_code == 200
