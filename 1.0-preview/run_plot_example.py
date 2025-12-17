#!/usr/bin/env python3
"""Visual README-style example that overlays the history and forecast."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from modeling import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams

IMAGES_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE_NAME = "synthetic_forecast.png"


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


def _quantile_array(quantiles: Dict[str | float, np.ndarray], target: float) -> np.ndarray | None:
    """Return the quantile array matching `target` regardless of key type."""
    for level, values in quantiles.items():
        try:
            if abs(float(level) - target) < 1e-6:
                return values
        except (TypeError, ValueError):
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the README synthetic series along with CiscoTsmMR forecasts."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator used to build synthetic data.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1280,
        help="Forecast horizon used when plotting projections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(); useful in headless environments when only saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Creating CiscoTsmMR model (this may download weights from Hugging Face)...")
    model = _create_model()

    series = _build_first_series(rng)
    print("Running forecast...")
    forecast = model.forecast(series, horizon_len=args.horizon)[0]
    mean_forecast = forecast["mean"]
    quantiles = forecast["quantiles"]
    p10 = _quantile_array(quantiles, 0.1)
    p90 = _quantile_array(quantiles, 0.9)

    history_index = np.arange(series.size)
    forecast_index = np.arange(series.size, series.size + mean_forecast.size)

    plt.figure(figsize=(14, 6))
    plt.plot(
        history_index,
        series,
        label="Input series (historical)",
        color="#1f77b4",
        linewidth=1.8,
    )
    plt.plot(
        forecast_index,
        mean_forecast,
        label="Forecast mean",
        color="#ff7f0e",
        linewidth=2.2,
    )

    if p10 is not None and p90 is not None:
        plt.fill_between(
            forecast_index,
            p10,
            p90,
            color="#ff7f0e",
            alpha=0.2,
            label="Forecast 10-90% band",
        )

    plt.axvline(
        history_index[-1],
        color="k",
        linestyle="--",
        linewidth=1.0,
        label="Forecast boundary",
    )
    plt.title("CiscoTsmMR Forecast vs. Input Series")
    plt.xlabel("Time step (minutes)")
    plt.ylabel("Synthetic metric value")
    plt.legend()
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    default_output = IMAGES_DIR / DEFAULT_IMAGE_NAME
    plt.savefig(default_output, dpi=200)
    print(f"Saved plot to {default_output}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
