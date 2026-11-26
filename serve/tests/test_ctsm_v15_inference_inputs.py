"""Regression checks for raw-history and explicit calendar validation."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from ctsm_1p5.modeling.cisco_tsm_mr import CiscoTsmMR


@pytest.fixture
def forecaster():
    model = object.__new__(CiscoTsmMR)
    model._device = torch.device("cpu")
    model._model = Mock()
    model._model.decode.return_value = [{}]
    return model


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array(["2025-01-06T10:00", "2025-01-06T10:02", "2025-01-06T10:05"]),
        pd.Series(
            pd.to_datetime(["2025-01-06T10:00", "2025-01-06T10:02", "2025-01-06T10:05"]),
            index=[11, 12, 13],
        ),
        [pd.to_datetime(["2025-01-06T10:00", "2025-01-06T10:02", "2025-01-06T10:05"])],
        np.array(["2025-01-06T10:00", "2025-01-06T10:02", "2025-01-06T10:05"], dtype="datetime64[s]").astype("int64"),
    ],
)
def test_explicit_clock_containers_preserve_observation_alignment(forecaster, timestamps):
    forecaster.forecast([[np.nan, 1.0, 2.0]], horizon_len=1, timestamps=timestamps)

    clock = forecaster._model.decode.call_args.kwargs["fine_tod"]
    weekdays = forecaster._model.decode.call_args.kwargs["fine_dow"]
    np.testing.assert_array_equal(clock[0, -2:, 0], [36120, 36300])
    np.testing.assert_array_equal(weekdays[0, -2:, 0], [0, 0])


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (["2025-01-06T10:00", "2025-01-06T10:01"], "2 timestamps for 3 observations"),
        (
            ["2025-01-06T10:00", "2025-01-06T10:01", "2025-01-06T10:02", "2025-01-06T10:03"],
            "4 timestamps for 3 observations",
        ),
        (["2025-01-06T10:00", "2025-01-06T10:01", "NaT"], "NaT"),
        (["2025-01-06T10:00", "2025-01-06T10:02", "2025-01-06T10:01"], "strictly increasing"),
        (
            ["2025-01-06T10:00:00", "2025-01-06T10:00:00.5", "2025-01-06T10:00:01"],
            "one-second resolution",
        ),
        (pd.date_range("2025-01-06", periods=3, freq="min", tz="UTC"), "timezone-naive"),
        (
            ["2025-01-06T10:00-08:00", "2025-01-06T10:01-08:00", "2025-01-06T10:02-08:00"],
            "timezone-naive",
        ),
    ],
)
def test_invalid_clock_is_rejected_before_model_execution(forecaster, timestamps, message):
    with pytest.raises(ValueError, match=message):
        forecaster.forecast([[np.nan, 1.0, 2.0]], horizon_len=1, timestamps=timestamps)

    forecaster._model.decode.assert_not_called()


@pytest.mark.parametrize("outlier", [1e40, -1e40, np.nextafter(np.float64(np.finfo(np.float32).max), np.inf)])
def test_finite_observation_outside_float32_range_is_rejected(forecaster, outlier):
    with pytest.raises(ValueError, match="finite observations outside the float32 range"):
        forecaster.forecast([[outlier, 1.0]], horizon_len=1, allow_missing_calendar=True)

    forecaster._model.decode.assert_not_called()


def test_actual_nonfinite_observations_still_follow_the_missing_data_policy(forecaster):
    forecaster.forecast([[np.nan, 1.0, np.inf, -np.inf, 2.0]], horizon_len=1, allow_missing_calendar=True)
    _, fine = forecaster._model.decode.call_args.args[0]
    assert torch.isfinite(fine).all()
