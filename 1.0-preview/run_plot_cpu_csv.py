#!/usr/bin/env python3
"""Plot CPU utilization sample data along with CiscoTsmMR forecasts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

from modeling import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams

DEFAULT_SAMPLE = Path(__file__).parent / "sample_data" / "cpu_utilization.csv"
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def _load_cpu_series(csv_path: Path) -> Tuple[list[datetime], np.ndarray]:
    """Load timestamps and values from a CPU utilization CSV."""
    timestamps: list[datetime] = []
    values: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "_time" not in reader.fieldnames or "cpu_util" not in reader.fieldnames:
            raise ValueError(
                f"Expected '_time' and 'cpu_util' columns in {csv_path}, got {reader.fieldnames}"
            )
        for row in reader:
            raw_time = row["_time"]
            raw_value = row["cpu_util"]
            if raw_time is None or raw_value is None:
                continue
            try:
                ts = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                try:
                    ts = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%S%z")
                except ValueError:
                    # Accept epoch seconds (int or float) as a fallback.
                    ts = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            timestamps.append(ts.astimezone(timezone.utc))
            values.append(float(raw_value))

    if not timestamps:
        raise ValueError(f"No rows parsed from {csv_path}")

    return timestamps, np.array(values, dtype=np.float32)


def _create_model() -> CiscoTsmMR:
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
    for level, values in quantiles.items():
        try:
            if abs(float(level) - target) < 1e-6:
                return values
        except (TypeError, ValueError):
            continue
    return None


def _build_forecast_timestamps(last_ts: datetime, horizon: int) -> Iterable[datetime]:
    for step in range(1, horizon + 1):
        yield last_ts + timedelta(minutes=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU utilization history from CSV along with CiscoTsmMR forecast."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="Path to a CSV containing '_time' and 'cpu_util' columns.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1280,
        help="Forecast horizon (in minutes) to request from the model.",
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
        help="Skip plt.show(); useful when running headless and only saving the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamps, values = _load_cpu_series(args.csv)
    series = values

    print("Creating CiscoTsmMR model (this may download weights from Hugging Face)...")
    model = _create_model()
    print(f"Running forecast for horizon={args.horizon} minutes...")
    forecast = model.forecast(series, horizon_len=args.horizon)[0]
    mean_forecast = forecast["mean"]
    quantiles = forecast["quantiles"]
    p10 = _quantile_array(quantiles, 0.1)
    p90 = _quantile_array(quantiles, 0.9)

    history_x = mdates.date2num(timestamps)
    forecast_times = list(_build_forecast_timestamps(timestamps[-1], mean_forecast.size))
    forecast_x = mdates.date2num(forecast_times)

    plt.figure(figsize=(14, 6))
    plt.plot(
        history_x,
        series,
        label="CPU utilization history",
        color="#1f77b4",
        linewidth=1.6,
    )
    plt.plot(
        forecast_x,
        mean_forecast,
        label="Forecast mean",
        color="#ff7f0e",
        linewidth=2.2,
    )

    if p10 is not None and p90 is not None:
        plt.fill_between(
            forecast_x,
            p10,
            p90,
            color="#ff7f0e",
            alpha=0.2,
            label="Forecast 10-90% band",
        )

    boundary = history_x[-1]
    plt.axvline(boundary, color="k", linestyle="--", linewidth=1.0, label="Forecast boundary")
    plt.title("CiscoTsmMR Forecast vs. CPU Utilization History")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("CPU Utilization [%]")
    plt.legend()
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    default_name = f"{args.csv.stem}_forecast.png"
    default_output = IMAGES_DIR / default_name
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
