"""Shared constants for the CDTSM inference server.

This module centralizes well-known, non-configurable values used across the
app (quantile labels, aliases, etc.) so they live in one place rather than
being embedded inline throughout feature modules.
"""

from __future__ import annotations

# API quantile name -> model output key. ``mean`` is a special sentinel.
QUANTILE_MAPPING: dict[str, str] = {
    "mean": "mean",
    "p1": "0.01",
    "p5": "0.05",
    "p10": "0.1",
    "p20": "0.2",
    "p25": "0.25",
    "p30": "0.3",
    "p40": "0.4",
    "p50": "0.5",
    "p60": "0.6",
    "p70": "0.7",
    "p75": "0.75",
    "p80": "0.8",
    "p90": "0.9",
    "p95": "0.95",
    "p99": "0.99",
}

# Aliases accepted from clients that send percentile-style labels.
QUANTILE_ALIASES: dict[str, str] = {
    "p01": "p1",
    "p001": "p1",
}

# Fixed by the CDTSM 1.0 checkpoint architecture.
CDTSM_NUM_LAYERS: int = 25

# Must stay in sync with :data:`QUANTILE_MAPPING` values.
CDTSM_DEFAULT_QUANTILES: list[float] = [
    0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99,
]
