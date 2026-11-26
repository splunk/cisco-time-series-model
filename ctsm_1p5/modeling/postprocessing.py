"""Quantile pooling, flip equivariance, and output transforms."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .constants import MEDIAN_QUANTILE_INDEX, QUANTILES


_Forecast = dict[str, Any]


def _trapezoid_weights(levels: ArrayLike) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64)
    return np.diff(np.r_[0.0, (levels[:-1] + levels[1:]) / 2, 1.0])


_POOL_WEIGHTS = np.outer(
    _trapezoid_weights((0.25, 0.5, 0.75)),
    _trapezoid_weights(QUANTILES),
).ravel()


def _complement_grid(quantiles: ArrayLike) -> tuple[np.ndarray, np.ndarray, int]:
    levels = np.asarray(quantiles)

    if levels.ndim != 1 or len(levels) != 15 or levels.dtype.kind not in 'fiu':
        raise ValueError('CTSM 1.5 requires 15 numeric quantiles')

    levels = levels.astype(np.float64)

    if (
        not np.isfinite(levels).all()
        or not np.all((levels > 0) & (levels < 1))
        or not np.all(np.diff(levels) > 0)
    ):
        raise ValueError('Quantiles must be finite, increasing, unique, and inside (0,1)')

    complement = []
    for q in levels:
        matches = np.flatnonzero(np.isclose(levels, 1 - q, rtol=0, atol=1e-7))
        if len(matches) != 1:
            raise ValueError('Every quantile needs one unambiguous complementary level')
        complement.append(int(matches[0]))

    median = np.flatnonzero(np.isclose(levels, .5, rtol=0, atol=1e-7))
    if len(median) != 1 or len(set(complement)) != len(levels):
        raise ValueError('Quantiles require one median and a complementary permutation')
    return levels, np.asarray(complement, dtype=np.intp), int(median[0])


def _reflected_inputs(histories: Iterable[ArrayLike]) -> list[np.ndarray]:
    # Cast before negation: signed integer minimums otherwise overflow.
    return [-np.asarray(row, dtype=np.float32) for row in histories]


def _quantile_arrays(
    prediction: _Forecast,
    levels: Sequence[float] | np.ndarray,
) -> tuple[list[str | float], list[np.ndarray]]:

    if not isinstance(prediction, dict) or not isinstance(prediction.get('quantiles'), dict):
        raise ValueError('Native forecasts must contain a quantiles dictionary')

    qmap = prediction['quantiles']
    if len(qmap) != len(levels):
        raise ValueError('Native forecasts must retain every one of the 15 quantiles')

    try:
        keys = list(qmap)
        numeric = np.asarray([float(k) for k in keys], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Native quantile keys must be numeric') from exc

    selected, arrays = [], []

    for q in levels:
        matches = np.flatnonzero(np.isclose(numeric, q, rtol=0, atol=1e-7))
        if len(matches) != 1:
            raise ValueError('Native quantile keys do not uniquely match the checkpoint grid')

        key = keys[int(matches[0])]
        values = np.asarray(qmap[key])
        if values.ndim != 1 or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('Native quantiles must be finite numeric horizon vectors')
        if arrays and values.shape != arrays[0].shape:
            raise ValueError('Native quantile horizons disagree')

        selected.append(key)
        arrays.append(values)
    return selected, arrays


def _stack_quantiles(prediction: _Forecast) -> tuple[list[str | float], np.ndarray]:
    keys, curves = _quantile_arrays(prediction, QUANTILES)
    return keys, np.stack(curves, axis=-1)


def _with_quantiles(
    prediction: _Forecast,
    keys: Sequence[str | float],
    matrix: np.ndarray,
) -> _Forecast:
    return {
        **prediction,
        "mean": matrix[:, MEDIAN_QUANTILE_INDEX].copy(),
        "quantiles": dict(zip(keys, matrix.T, strict=True)),
    }


def _weighted_quantiles(samples: np.ndarray) -> np.ndarray:
    """Invert a weighted CDF using the midpoint of each sample's mass."""
    order = np.argsort(samples, axis=1)
    values = np.take_along_axis(samples, order, axis=1)
    weights = _POOL_WEIGHTS[order]
    midpoints = (np.cumsum(weights, axis=1) - weights / 2) / weights.sum(axis=1, keepdims=True)
    result = np.empty((len(values), len(QUANTILES)), dtype=np.float64)
    rows = np.arange(len(values))

    for index, level in enumerate(QUANTILES):
        upper = np.clip((midpoints < level).sum(axis=1), 1, values.shape[1] - 1)
        lower = upper - 1
        c0, c1 = midpoints[rows, lower], midpoints[rows, upper]
        v0, v1 = values[rows, lower], values[rows, upper]
        fraction = np.clip((level - c0) / np.maximum(c1 - c0, 1e-12), 0.0, 1.0)
        result[:, index] = v0 + fraction * (v1 - v0)
    return result


def pool_three_paths(
    lower: Sequence[_Forecast],
    median: Sequence[_Forecast],
    upper: Sequence[_Forecast],
    output_patch_length: int,
) -> tuple[list[_Forecast], list[_Forecast]]:
    """Return the native first block and the pooled suffix for one sign."""
    if not (len(lower) == len(median) == len(upper)):
        raise ValueError("The three paths returned different batch sizes")

    median_keys, middle = zip(*map(_stack_quantiles, median), strict=True)
    other_paths = (
        [_stack_quantiles(prediction)[1] for prediction in lower],
        [_stack_quantiles(prediction)[1] for prediction in upper],
    )
    paths = [
        np.stack([matrix[output_patch_length:] for matrix in forecasts])
        for forecasts in (other_paths[0], middle, other_paths[1])
    ]

    if any(path.shape != paths[1].shape for path in paths):
        raise ValueError("The three paths returned different forecast horizons")

    sorted_paths = [np.sort(path, axis=-1).reshape(-1, len(QUANTILES)) for path in paths]
    samples = np.concatenate(sorted_paths, axis=1).astype(np.float64, copy=False)
    pooled = _weighted_quantiles(samples).reshape(paths[1].shape)

    prefix = [
        _with_quantiles(prediction, keys, matrix[:output_patch_length])
        for prediction, keys, matrix in zip(median, median_keys, middle, strict=True)
    ]
    suffix = [
        _with_quantiles(prediction, keys, matrix)
        for prediction, keys, matrix in zip(median, median_keys, pooled, strict=True)
    ]

    return prefix, suffix


def join_forecast_blocks(prefix: Iterable[_Forecast], suffix: Iterable[_Forecast]) -> list[_Forecast]:
    """Join the native prefix and pooled suffix after optional flip equivariance."""
    predictions = []

    for first, later in zip(prefix, suffix, strict=True):
        keys, first_matrix = _stack_quantiles(first)
        _, later_matrix = _stack_quantiles(later)
        matrix = np.concatenate((first_matrix, later_matrix), axis=0)
        predictions.append(_with_quantiles(first, keys, matrix))

    return predictions


def sort_quantiles(predictions: Iterable[_Forecast]) -> list[_Forecast]:
    """Sort all quantiles at each forecast step and return the resulting P50."""
    sorted_predictions = []

    for prediction in predictions:
        keys, matrix = _stack_quantiles(prediction)
        sorted_predictions.append(_with_quantiles(prediction, keys, np.sort(matrix, axis=-1)))

    return sorted_predictions


def _merge_symmetric(
    positive: Sequence[_Forecast],
    reflected: Sequence[_Forecast],
    levels: Sequence[float] | np.ndarray,
    complements: Sequence[int] | np.ndarray,
    median: int,
) -> list[_Forecast]:
    if len(positive) != len(reflected):
        raise ValueError('The two sign orientations returned different batch sizes')

    result = []
    for left, right in zip(positive, reflected):
        keys, original = _quantile_arrays(left, levels)
        _, mirrored = _quantile_arrays(right, levels)
        curves = []
        for i, values in enumerate(original):
            other = mirrored[int(complements[i])]
            if values.shape != other.shape:
                raise ValueError('The two sign orientations returned different horizons')

            dtype = np.result_type(values.dtype, other.dtype, np.float32)
            # Halve first to avoid avoidable float32 overflow in (left-right)/2.
            merged = values.astype(dtype, copy=False) * .5 - other.astype(dtype, copy=False) * .5
            if not np.isfinite(merged).all():
                raise FloatingPointError('Flip equivariance produced a nonfinite forecast')
            curves.append(merged)

        row = dict(left)
        row['quantiles'] = dict(zip(keys, curves))
        row['mean'] = curves[median].copy()
        result.append(row)

    return result


def gates(values: ArrayLike) -> tuple[bool, bool]:
    x = np.asarray(values)
    assert x.ndim == 1, 'Inputs must be univariate raw histories'

    finite = x[np.isfinite(x)].astype(np.float64)

    return (
        bool(len(finite) and np.all(finite >= 0)),
        bool(len(finite) and np.all(np.abs(finite - np.rint(finite)) <= 1e-6)),
    )


def transform_forecast(prediction: _Forecast, nonnegative: bool, integer: bool) -> _Forecast:
    def apply(values: ArrayLike) -> np.ndarray:
        x = np.asarray(values)
        y = np.maximum(x, 0) if nonnegative else x.copy()
        y = np.rint(y) if integer else y

        if y.shape != x.shape:
            raise ValueError('Clip/snap changed the forecast shape')

        if not np.isfinite(y).all():
            raise FloatingPointError('Clip/snap produced a nonfinite forecast')

        return y

    result = dict(prediction)
    result['quantiles'] = {k: apply(v) for k, v in prediction['quantiles'].items()}

    if 'mean' in prediction:
        result['mean'] = apply(prediction['mean'])

    return result
