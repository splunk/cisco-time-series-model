from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import Settings, get_settings
from app.core.exceptions import BadInputError, ModelNotReadyError
from app.core.quantiles import extract_from_forecast_dict, resolve_requested_quantiles
from app.models.schemas import InferRequestBody, PredictionItem
from app.services.cisco_adapter import build_cisco_tsm_model, forecast_supports_restrict_quantiles

if TYPE_CHECKING:
    from cisco_tsm import CiscoTsmMR

logger = logging.getLogger(__name__)

ModelLoadPhase = Literal["not_started", "in_progress", "succeeded", "failed"]


class ModelService:
    """Singleton-style service: one loaded model per process."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._model: CiscoTsmMR | None = None
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
            self._model, self._inference_backend = build_cisco_tsm_model(self._settings)
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

        if len(raw) != len(body.payload):
            raise BadInputError(
                "Unexpected forecast batch size",
                details={"expected": len(body.payload), "got": len(raw)},
            )

        predictions: list[PredictionItem] = []
        for fc in raw:
            mean_list, q_dict = extract_from_forecast_dict(fc, requested)
            predictions.append(PredictionItem(mean=mean_list, quantiles=q_dict))
        return predictions


_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
        _model_service.configure_from_settings(get_settings())
    return _model_service


def reset_model_service_for_tests() -> None:
    global _model_service
    _model_service = None
