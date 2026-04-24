"""Map API quantile labels (p40, mean, …) to Cisco TSM output keys."""

from __future__ import annotations

from typing import Iterable

from app.core.constants import QUANTILE_ALIASES, QUANTILE_MAPPING
from app.core.exceptions import BadInputError

__all__ = [
    "QUANTILE_MAPPING",
    "extract_from_forecast_dict",
    "model_key_for_api_name",
    "normalize_quantile_name",
    "resolve_requested_quantiles",
]


def normalize_quantile_name(name: str) -> str:
    key = name.strip().lower()
    return QUANTILE_ALIASES.get(key, key)


def resolve_requested_quantiles(requested: Iterable[str]) -> list[str]:
    items = list(requested)
    if not items:
        return list(QUANTILE_MAPPING.keys())
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        q = normalize_quantile_name(raw)
        if q not in QUANTILE_MAPPING:
            supported = ", ".join(sorted(QUANTILE_MAPPING))
            raise BadInputError(
                f"Unsupported quantile {raw!r}. Supported: {supported}",
                details={"quantile": raw, "supported": sorted(QUANTILE_MAPPING.keys())},
            )
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def model_key_for_api_name(api_name: str) -> str:
    return QUANTILE_MAPPING[normalize_quantile_name(api_name)]


def extract_from_forecast_dict(
    forecast: dict,
    requested_api_names: list[str],
) -> tuple[list[float | None], dict[str, list[float | None]]]:
    """Build mean + quantiles dict for HTTP response from one model output dict."""
    mean_arr = forecast.get("mean")
    if mean_arr is None:
        raise BadInputError("Model output missing 'mean'")
    mean_list = [None if x is None else float(x) for x in mean_arr.tolist()]

    raw_q: dict = forecast.get("quantiles") or {}
    normalized_model_q: dict[str, object] = {}
    for k, v in raw_q.items():
        normalized_model_q[_normalize_model_quantile_key(k)] = v

    quantiles_out: dict[str, list[float]] = {}
    for api_name in requested_api_names:
        if api_name == "mean":
            continue
        mk = model_key_for_api_name(api_name)
        arr = normalized_model_q.get(mk)
        if arr is None:
            raise BadInputError(
                f"Model output missing quantile key {mk!r} (for {api_name!r})",
                details={"model_key": mk, "api_name": api_name, "available": sorted(normalized_model_q)},
            )
        quantiles_out[api_name] = [None if x is None else float(x) for x in arr.tolist()]

    return mean_list, quantiles_out


def _normalize_model_quantile_key(key: object) -> str:
    if isinstance(key, float):
        if key == int(key):
            return str(int(key))
        s = f"{key:.2f}".rstrip("0").rstrip(".")
        return s
    s = str(key).strip()
    if s.replace(".", "", 1).isdigit():
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
        except ValueError:
            pass
    return s
