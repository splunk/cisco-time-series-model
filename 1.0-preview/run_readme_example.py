#!/usr/bin/env python3
"""Reproduce the README example with a single command."""

from __future__ import annotations

import argparse
from typing import Dict, Sequence

import numpy as np
import torch

from modeling import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams


def _build_first_series(rng: np.random.Generator) -> np.ndarray:
    """Create the synthetic series described in the README example."""
    t = np.arange(512 * 60, dtype=np.float32)
    hours = (t.size + 59) // 60
    k = np.arange(hours, dtype=np.float32)
    hourly_base = (80 + 0.1 * k) * (1 + 0.25 * np.sin(2 * np.pi * k / 24))
    noisy_minute_scale = 1 + 0.05 * np.sin(2 * np.pi * t / 30)
    return hourly_base[(t // 60).astype(int)] * noisy_minute_scale + rng.normal(
        0, 0.4, size=t.size
    )


def _build_second_series(rng: np.random.Generator) -> np.ndarray:
    """Create the auxiliary series used in the multi-series README example."""
    t = np.arange(25_000, dtype=np.float32)
    hours = (t.size + 59) // 60
    k = np.arange(hours, dtype=np.float32)
    hourly_trend = 120 / (1 + np.exp(-0.01 * (k - 300))) + 10 * np.cos(
        2 * np.pi * k / (24 * 7)
    )
    minute_pattern = 2 * np.sin(2 * np.pi * t / 60)
    return hourly_trend[(t // 60).astype(int)] + minute_pattern + rng.normal(
        0, 0.5, size=t.size
    )


def _summarize_forecasts(
    label: str, forecasts: Sequence[Dict[str, np.ndarray]]
) -> None:
    def _quantile_sort_key(level: str) -> tuple[int, float | str]:
        try:
            return (0, float(level))
        except (TypeError, ValueError):
            return (1, str(level))

    for idx, output in enumerate(forecasts):
        mean = output["mean"]
        quantiles = output["quantiles"]
        quantile_levels = sorted(quantiles.keys(), key=_quantile_sort_key)
        median_level = quantile_levels[len(quantile_levels) // 2]
        try:
            median_level_fmt = f"{float(median_level):.1f}"
        except (TypeError, ValueError):
            median_level_fmt = str(median_level)
        print(
            f"[{label} series {idx}] "
            f"mean shape={mean.shape}, first 5={np.round(mean[:5], 3)}"
        )
        print(
            f"  median quantile ({median_level_fmt}) first 5="
            f"{np.round(quantiles[median_level][:5], 3)}"
        )


def _create_model() -> CiscoTsmMR:
    """Instantiate the CiscoTsmMR model exactly as in the README."""
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    hparams = TimesFmHparams(
        num_layers=50,
        use_positional_embedding=False,
        backend=backend,
    )
    checkpoint = TimesFmCheckpoint(
        huggingface_repo_id="cisco-ai/cisco-time-series-model-1.0-preview"
    )
    return CiscoTsmMR(
        hparams=hparams,
        checkpoint=checkpoint,
        use_resolution_embeddings=True,
        use_special_token=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the README example forecasts with CiscoTsmMR."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator used to build synthetic data.",
    )
    parser.add_argument(
        "--short-horizon",
        type=int,
        default=128,
        help="Forecast horizon used for the single and multi-series examples.",
    )
    parser.add_argument(
        "--long-horizon",
        type=int,
        default=240,
        help="Forecast horizon used in the long-horizon example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Creating CiscoTsmMR model (this may download weights from Hugging Face)...")
    model = _create_model()

    series_one = _build_first_series(rng)
    series_two = _build_second_series(rng)

    print("\nRunning single-series forecast (README example)...")
    single_series_forecast = model.forecast(series_one, horizon_len=args.short_horizon)
    _summarize_forecasts("single", single_series_forecast)

    print("\nRunning multi-series forecast (README example)...")
    multi_series_forecast = model.forecast(
        [series_one, series_two], horizon_len=args.short_horizon
    )
    _summarize_forecasts("multi", multi_series_forecast)

    print("\nRunning long-horizon forecast (README example)...")
    long_horizon_forecast = model.forecast(series_one, horizon_len=args.long_horizon)
    _summarize_forecasts("long", long_horizon_forecast)

    print("\nDone! Review the summaries above for quick sanity checks.")


if __name__ == "__main__":
    main()
