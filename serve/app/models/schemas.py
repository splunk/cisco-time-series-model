from __future__ import annotations

from typing import Any, Literal

from cisco_tsm.constants import Constants
from pydantic import BaseModel, Field, field_validator

MAX_COARSE: int = int(Constants.CONTEXT_LEN_COARSE.value)
MAX_FINE: int = int(Constants.CONTEXT_LEN_FINE.value)


class SeriesPayload(BaseModel):
    coarse_ctx: list[float] = Field(..., min_length=1)
    fine_ctx: list[float] = Field(..., min_length=1)

    @field_validator("coarse_ctx", "fine_ctx", mode="before")
    @classmethod
    def coerce_numeric(cls, v: object) -> list[float]:
        """Coerce each element to ``float``; map JSON ``null`` to ``NaN``."""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("must be a non-empty list")
        out: list[float] = []
        for i, x in enumerate(v):
            if x is None:
                out.append(float("nan"))
                continue
            try:
                out.append(float(x))
            except (TypeError, ValueError) as e:
                raise ValueError(f"non-numeric value at index {i}") from e
        return out


class InferMetadata(BaseModel):
    quantiles: list[str] = Field(
        default_factory=lambda: ["mean", "p50"],
        description="Requested outputs: mean and/or p10, p50, …",
    )


class InferRequestBody(BaseModel):
    payload: list[SeriesPayload]
    model: Literal["CDTSM"] = "CDTSM"
    metadata: InferMetadata = Field(default_factory=InferMetadata)

    @field_validator("payload")
    @classmethod
    def non_empty_payload(cls, v: list[SeriesPayload]) -> list[SeriesPayload]:
        if not v:
            raise ValueError("payload must be non-empty")
        return v


class PredictionItem(BaseModel):
    mean: list[float]
    quantiles: dict[str, list[float]]


class InferSuccessResponse(BaseModel):
    request_id: str
    model: str
    horizon: int
    predictions: list[PredictionItem]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class InferErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
