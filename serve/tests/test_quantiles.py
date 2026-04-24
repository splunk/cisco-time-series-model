from __future__ import annotations

import numpy as np
import pytest

from app.core.constants import QUANTILE_MAPPING
from app.core.exceptions import BadInputError
from app.core.quantiles import (
    extract_from_forecast_dict,
    model_key_for_api_name,
    normalize_quantile_name,
    resolve_requested_quantiles,
)


def test_quantile_mapping_keys():
    assert QUANTILE_MAPPING["p40"] == "0.4"
    assert QUANTILE_MAPPING["p90"] == "0.9"
    assert QUANTILE_MAPPING["mean"] == "mean"


def test_normalize_quantile_name():
    assert normalize_quantile_name("P40") == "p40"
    assert normalize_quantile_name("p01") == "p1"


def test_model_key_for_api_name():
    assert model_key_for_api_name("p40") == "0.4"


def test_resolve_requested_quantiles_rejects_unknown():
    with pytest.raises(BadInputError) as e:
        resolve_requested_quantiles(["p41"])
    assert "Unsupported quantile" in str(e.value)


def test_resolve_requested_quantiles_dedupes():
    assert resolve_requested_quantiles(["p50", "p50", "mean"]) == ["p50", "mean"]


def test_extract_from_forecast_dict_maps_response_labels():
    horizon = 3
    forecast = {
        "mean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "quantiles": {
            "0.4": np.array([1.1, 2.1, 3.1], dtype=np.float32),
            0.9: np.array([1.9, 2.9, 3.9], dtype=np.float32),
        },
    }
    mean, q = extract_from_forecast_dict(forecast, ["mean", "p40", "p90"])
    assert mean == pytest.approx([1.0, 2.0, 3.0])
    assert list(q.keys()) == ["p40", "p90"]
    assert q["p40"] == pytest.approx([1.1, 2.1, 3.1])
    assert q["p90"] == pytest.approx([1.9, 2.9, 3.9])
