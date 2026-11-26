from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import Settings, get_settings
from app.core.exceptions import BadInputError, ModelNotReadyError
from app.core.quantiles import extract_from_forecast_dict, resolve_requested_quantiles
from app.models.schemas import InferRequestBody, InferRequestBodyV15, PredictionItem, SeriesPayloadV15
from app.services.cisco_adapter import (
    build_cisco_tsm_model,
    build_ctsm_v15_model,
    forecast_supports_restrict_quantiles,
)

if TYPE_CHECKING:
    from cisco_tsm import CiscoTsmMR
    from ctsm_1p5.modeling import CTSMForecaster

logger = logging.getLogger(__name__)

ModelLoadPhase = Literal["not_started", "in_progress", "succeeded", "failed"]


class ModelService:
    """Manage one model and its independent load state."""

    def __init__(self, *, v15: bool = False) -> None:
        self._v15 = v15
        self._settings: Settings | None = None
        self._model: CiscoTsmMR | CTSMForecaster | None = None
        self._load_error: str | None = None
        self._inference_backend: str | None = None
        self._model_load_phase: ModelLoadPhase = "not_started"
        self._load_started_at: datetime | None = None
        self._load_completed_at: datetime | None = None

    def configure_from_settings(self, settings: Settings) -> None:
        self._settings = settings

    def load(self) -> None:
        if self._settings is None:
            raise RuntimeError("ModelService.configure_from_settings() must be called before load()")
        self._load_started_at = datetime.now(timezone.utc)
        self._model_load_phase = "in_progress"
        self._load_completed_at = None
        logger.info("Model load started")

        if self._settings.serving_backend != "native":
            self._load_error = (
                f"Serving backend {self._settings.serving_backend!r} is not supported; "
                "CDTSM only runs as an in-process ``native`` backend. "
                "Set CDTSM_SERVING_BACKEND=native (the default) to proceed."
            )
            self._model_load_phase = "failed"
            self._load_completed_at = datetime.now(timezone.utc)
            logger.error("Model load failed: %s", self._load_error)
            return

        try:
            build_model = build_ctsm_v15_model if self._v15 else build_cisco_tsm_model
            self._model, self._inference_backend = build_model(self._settings)
            self._load_error = None
            self._model_load_phase = "succeeded"
            self._load_completed_at = datetime.now(timezone.utc)
            logger.info(
                "Model load finished successfully",
                extra={
                    "structured": {
                        "inference_backend": self._inference_backend,
                        "seconds": self.load_elapsed_seconds,
                    }
                },
            )
        except Exception as e:
            self._load_error = str(e)
            self._model = None
            self._inference_backend = None
            self._model_load_phase = "failed"
            self._load_completed_at = datetime.now(timezone.utc)
            logger.exception("Model load failed", extra={"structured": {"error": self._load_error}})

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def inference_backend(self) -> str | None:
        return self._inference_backend

    @property
    def load_elapsed_seconds(self) -> float | None:
        if self._load_started_at is None:
            return None
        end = self._load_completed_at or datetime.now(timezone.utc)
        return round((end - self._load_started_at).total_seconds(), 3)

    def model_load_status_dict(self) -> dict[str, Any]:
        return {
            "phase": self._model_load_phase,
            "started_at": self._load_started_at.isoformat() if self._load_started_at else None,
            "completed_at": self._load_completed_at.isoformat() if self._load_completed_at else None,
            "elapsed_seconds": self.load_elapsed_seconds,
        }

    def infer(self, body: InferRequestBody, horizon: int) -> list[PredictionItem]:
        if self._settings is None or self._model is None:
            raise ModelNotReadyError("Inference engine not initialized")
        if body.model != "CDTSM":
            raise BadInputError("Only model 'CDTSM' is supported", details={"model": body.model})

        requested = resolve_requested_quantiles(body.metadata.quantiles)
        if "mean" not in requested:
            requested = ["mean", *requested]

        batch_pairs: list[tuple[list[float], list[float]]] = [
            (item.coarse_ctx, item.fine_ctx) for item in body.payload
        ]

        forecast_kwargs: dict[str, Any] = {
            "horizon_len": horizon,
            "batch_size": self._settings.infer_batch_size,
        }
        if forecast_supports_restrict_quantiles():
            forecast_kwargs["restrict_quantiles"] = True

        raw: list[dict[str, Any]] = self._model.forecast(batch_pairs, **forecast_kwargs)

        return _format_predictions(raw, requested, len(body.payload))

    def infer_v15(self, body: InferRequestBodyV15, horizon: int) -> list[PredictionItem]:
        if self._settings is None or self._model is None:
            raise ModelNotReadyError("Inference engine not initialized")

        requested = resolve_requested_quantiles(body.metadata.quantiles)
        groups: dict[bool, list[tuple[int, list[float], list[datetime] | None]]] = {
            False: [], True: [],
        }
        for index, item in enumerate(body.payload):
            clock = _parse_v15_timestamps(item, index)
            values = [float("nan") if value is None else value for value in item.history]
            groups[clock is not None].append((index, values, clock))

        predictions: dict[int, PredictionItem] = {}
        for has_timestamps, batch in groups.items():
            if not batch:
                continue
            indices, histories, clocks = zip(*batch)
            try:
                raw = self._model.forecast(
                    list(histories),
                    horizon_len=horizon,
                    batch_size=self._settings.infer_batch_size,
                    timestamps=list(clocks) if has_timestamps else None,
                )
            except ValueError as exc:
                raise BadInputError(str(exc)) from exc
            predictions.update(zip(indices, _format_predictions(raw, requested, len(batch))))
        return [predictions[index] for index in range(len(body.payload))]


def _format_predictions(raw: list[dict[str, Any]], requested: list[str], expected: int) -> list[PredictionItem]:
    if len(raw) != expected:
        raise BadInputError(
            "Unexpected forecast batch size",
            details={"expected": expected, "got": len(raw)},
        )

    predictions: list[PredictionItem] = []
    for fc in raw:
        mean_list, q_dict = extract_from_forecast_dict(fc, requested)
        predictions.append(PredictionItem(mean=mean_list, quantiles=q_dict))
    return predictions


def _parse_v15_timestamps(item: SeriesPayloadV15, index: int) -> list[datetime] | None:
    if item.timestamps is None:
        return None

    details = {"series_index": index}
    if len(item.timestamps) != len(item.history):
        raise BadInputError("Supply one timestamp per observation", details=details)
    if len(item.timestamps) < 2:
        raise BadInputError("Timestamped history needs at least two observations", details=details)
    try:
        clock = [datetime.fromisoformat(value) for value in item.timestamps]
    except ValueError as exc:
        raise BadInputError("Timestamps must be ISO dates or datetimes", details=details) from exc
    if any(value.tzinfo is not None for value in clock):
        raise BadInputError("Timestamps must be timezone-naive", details=details)
    if any(left >= right for left, right in zip(clock, clock[1:])):
        raise BadInputError("Timestamps must be strictly increasing", details=details)
    return clock


_model_service: ModelService | None = None
_v15_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
        _model_service.configure_from_settings(get_settings())
    return _model_service


def get_v15_model_service() -> ModelService:
    global _v15_model_service
    if _v15_model_service is None:
        _v15_model_service = ModelService(v15=True)
        _v15_model_service.configure_from_settings(get_settings())
    return _v15_model_service


def reset_model_service_for_tests() -> None:
    global _model_service, _v15_model_service
    _model_service = None
    _v15_model_service = None
