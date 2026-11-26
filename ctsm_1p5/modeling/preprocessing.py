"""Missing-value handling for raw observation histories."""

import numpy as np


def trim_leading_missing(history: np.ndarray) -> np.ndarray:
    """Start the history at its first observation, or keep all missing values."""
    if history.size == 0:
        raise ValueError("cannot trim an empty history")

    if not np.isnan(history[0]):
        return history[:]

    observed = np.flatnonzero(~np.isnan(history))
    if observed.size == 0:
        return history[:]

    return history[observed[0]:]


def interpolate_missing(history: np.ndarray) -> np.ndarray:
    """Interpolate gaps in place and use zeros for an entirely missing history."""
    missing = np.isnan(history)

    if not missing.any():
        return history

    if missing.all():
        return np.zeros_like(history)

    gaps = np.flatnonzero(missing)
    samples = np.flatnonzero(~missing)
    history[gaps] = np.interp(gaps, samples, history[samples])

    return history
