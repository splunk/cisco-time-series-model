from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.models.schemas import InferRequestBody, SeriesPayload


def test_series_accepts_nan_and_inf():
    """Non-finite values are passed through unchanged."""
    s = SeriesPayload(
        coarse_ctx=[1.0, float("nan"), float("inf"), float("-inf"), 5.0],
        fine_ctx=[1.0],
    )
    assert math.isnan(s.coarse_ctx[1])
    assert math.isinf(s.coarse_ctx[2]) and s.coarse_ctx[2] > 0
    assert math.isinf(s.coarse_ctx[3]) and s.coarse_ctx[3] < 0
    assert s.coarse_ctx[0] == 1.0 and s.coarse_ctx[4] == 5.0


def test_series_accepts_json_null_as_nan():
    """JSON ``null`` is coerced to ``NaN``."""
    s = SeriesPayload.model_validate({"coarse_ctx": [1.0, None, 3.0], "fine_ctx": [1.0]})
    assert s.coarse_ctx[0] == 1.0
    assert math.isnan(s.coarse_ctx[1])
    assert s.coarse_ctx[2] == 3.0


def test_series_rejects_non_numeric():
    with pytest.raises(ValidationError) as e:
        SeriesPayload(coarse_ctx=[1.0, "abc"], fine_ctx=[1.0])  # type: ignore[list-item]
    assert "non-numeric" in str(e.value).lower()


def test_series_rejects_empty():
    with pytest.raises(ValidationError):
        SeriesPayload(coarse_ctx=[], fine_ctx=[1.0])


def test_infer_body_rejects_empty_payload():
    with pytest.raises(ValidationError):
        InferRequestBody(payload=[], model="CDTSM")


def test_infer_body_accepts_contract():
    body = InferRequestBody.model_validate(
        {
            "payload": [
                {"coarse_ctx": [1.0, 2.0], "fine_ctx": [3.0]},
            ],
            "model": "CDTSM",
            "metadata": {"quantiles": ["mean", "p40"]},
        }
    )
    assert len(body.payload) == 1
