#!/usr/bin/env python3
"""Minimal Flask application exposing CiscoTsmMR CPU forecast plotting as a web service."""

from __future__ import annotations

import base64
import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from run_plot_cpu_csv import (
    DEFAULT_SAMPLE,
    _build_forecast_timestamps,
    _create_model,
    _quantile_array,
    _load_cpu_series,
)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

DEFAULT_HORIZON = 1280
MAX_HORIZON = 524288


@dataclass
class SeriesPayload:
    timestamps: list[datetime]
    values: np.ndarray


@lru_cache(maxsize=1)
def _model_singleton():
    return _create_model()


def _parse_timestamp(raw: str) -> datetime:
    """Parse timestamps similarly to run_plot_cpu_csv."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    return datetime.fromtimestamp(float(raw), tz=timezone.utc)


def _load_series_from_text(text: str) -> SeriesPayload:
    if not text or not text.strip():
        raise ValueError("CSV text is empty.")
    timestamps: list[datetime] = []
    values: list[float] = []
    reader = csv.DictReader(io.StringIO(text))
    if "_time" not in reader.fieldnames or "cpu_util" not in reader.fieldnames:
        raise ValueError("CSV must contain '_time' and 'cpu_util' columns.")
    for row in reader:
        raw_time = row.get("_time")
        raw_value = row.get("cpu_util")
        if raw_time is None or raw_value is None:
            continue
        ts = _parse_timestamp(raw_time)
        timestamps.append(ts)
        values.append(float(raw_value))
    if not timestamps:
        raise ValueError("No valid rows found in CSV data.")
    return SeriesPayload(timestamps, np.array(values, dtype=np.float32))


def _resolve_series(csv_text: Optional[str], file_storage, use_sample: bool) -> SeriesPayload:
    if file_storage and file_storage.filename:
        data = file_storage.read().decode("utf-8")
        return _load_series_from_text(data)
    if csv_text and csv_text.strip():
        return _load_series_from_text(csv_text)
    if use_sample or not (csv_text or file_storage):
        timestamps, values = _load_cpu_series(DEFAULT_SAMPLE)
        return SeriesPayload(timestamps, values)
    raise ValueError("No CSV data provided and sample usage disabled.")


def _render_plot(timestamps: list[datetime], series: np.ndarray, horizon: int) -> Tuple[bytes, dict]:
    model = _model_singleton()
    forecast = model.forecast(series, horizon_len=horizon)[0]
    mean_forecast = forecast["mean"]
    quantiles = forecast["quantiles"]
    p10 = _quantile_array(quantiles, 0.1)
    p90 = _quantile_array(quantiles, 0.9)

    history_x = mdates.date2num(timestamps)
    forecast_times = list(_build_forecast_timestamps(timestamps[-1], mean_forecast.size))
    forecast_x = mdates.date2num(forecast_times)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history_x, series, label="CPU utilization history", color="#1f77b4", linewidth=1.6)
    ax.plot(forecast_x, mean_forecast, label="Forecast mean", color="#ff7f0e", linewidth=2.2)
    if p10 is not None and p90 is not None:
        ax.fill_between(forecast_x, p10, p90, color="#ff7f0e", alpha=0.2, label="Forecast 10-90% band")

    boundary = history_x[-1]
    ax.axvline(boundary, color="k", linestyle="--", linewidth=1.0, label="Forecast boundary")
    ax.set_title("CiscoTsmMR Forecast vs. CPU Utilization History")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("CPU Utilization [%]")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\\n%H:%M", tz=timezone.utc))
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)
    buffer.seek(0)

    history_timestamps = [ts.isoformat() for ts in timestamps]
    forecast_timestamps = [ts.isoformat() for ts in forecast_times]
    plot_payload = {
        "history": {
            "timestamps": history_timestamps,
            "values": series.astype(float).tolist(),
        },
        "forecast": {
            "timestamps": forecast_timestamps,
            "mean": mean_forecast.astype(float).tolist(),
            "p10": p10.astype(float).tolist() if p10 is not None else None,
            "p90": p90.astype(float).tolist() if p90 is not None else None,
        },
        "boundary_timestamp": timestamps[-1].isoformat(),
    }

    return buffer.read(), plot_payload


def _extract_params(req) -> Tuple[int, SeriesPayload]:
    horizon_raw = DEFAULT_HORIZON
    csv_text = None
    use_sample = False
    if req.is_json:
        data = req.get_json(silent=True) or {}
        csv_text = data.get("csv_data")
        horizon_raw = data.get("horizon", DEFAULT_HORIZON)
        use_sample = bool(data.get("use_sample", False))
    else:
        form = req.form
        csv_text = form.get("csv_text")
        horizon_raw = form.get("horizon", DEFAULT_HORIZON)
        use_sample = form.get("use_sample") in ("1", "true", "on")
    try:
        horizon = int(horizon_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Horizon must be an integer.") from exc
    if horizon < 1 or horizon > MAX_HORIZON:
        raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON}.")
    series_payload = _resolve_series(csv_text, req.files.get("csv_file"), use_sample)
    return horizon, series_payload


def _sample_preview() -> str:
    sample_lines = DEFAULT_SAMPLE.read_text(encoding="utf-8").splitlines()[:20]
    return "\n".join(sample_lines)


SAMPLE_PREVIEW = _sample_preview()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template(
            "cpu_forecast.html",
            error=None,
            image_data=None,
            csv_text=SAMPLE_PREVIEW,
            horizon=DEFAULT_HORIZON,
            use_sample=True,
            max_horizon=MAX_HORIZON,
            plot_payload=None,
        )
    try:
        horizon, payload = _extract_params(request)
        png_bytes, plot_payload = _render_plot(payload.timestamps, payload.values, horizon)
        img_b64 = base64.b64encode(png_bytes).decode("ascii")
        return render_template(
            "cpu_forecast.html",
            error=None,
            image_data=img_b64,
            csv_text=request.form.get("csv_text", SAMPLE_PREVIEW),
            horizon=horizon,
            use_sample=request.form.get("use_sample") in ("1", "true", "on"),
            max_horizon=MAX_HORIZON,
            plot_payload=plot_payload,
        )
    except ValueError as exc:
        return (
            render_template(
                "cpu_forecast.html",
                error=str(exc),
                image_data=None,
                csv_text=request.form.get("csv_text", SAMPLE_PREVIEW),
                horizon=request.form.get("horizon", DEFAULT_HORIZON),
                use_sample=request.form.get("use_sample") in ("1", "true", "on"),
                max_horizon=MAX_HORIZON,
                plot_payload=None,
            ),
            400,
        )


@app.post("/api/forecast")
def api_forecast():
    try:
        horizon, payload = _extract_params(request)
        png_bytes, _ = _render_plot(payload.timestamps, payload.values, horizon)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=True,
        download_name="cpu_forecast.png",
    )


def main():
    port = int(os.environ.get("CPU_FORECAST_SERVER_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
