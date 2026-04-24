from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelNotReadyError
from app.models.schemas import (
    InferErrorResponse,
    InferRequestBody,
    InferSuccessResponse,
)
from app.services.auth import auth_scheme, verify_auth_token
from app.services.model_service import ModelService, get_model_service

router = APIRouter()


@router.get("/")
def root(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    return {
        "service": settings.service_name,
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "infer": "POST /cdtsm/v1/ai/infer",
            "docs": "/docs",
        },
    }


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    return {"status": "ok", "service": settings.service_name}


@router.get("/ready", response_model=None)
def ready(
    svc: Annotated[ModelService, Depends(get_model_service)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    if not svc.is_ready:
        body: dict = {
            "status": "not_ready",
            "service": settings.service_name,
            "load_error": svc.load_error,
            "model_load": svc.model_load_status_dict(),
        }
        if svc.load_error:
            body["message"] = "Model load failed; see load_error."
        else:
            body["message"] = "Model is loading. Poll GET /ready until ready."
        return JSONResponse(status_code=503, content=body)
    return {
        "status": "ready",
        "service": settings.service_name,
        "torch_backend": settings.torch_backend,
        "inference_backend": svc.inference_backend,
        "model_load": svc.model_load_status_dict(),
    }


@router.post(
    "/cdtsm/v1/ai/infer",
    response_model=InferSuccessResponse,
    responses={400: {"model": InferErrorResponse}, 401: {"model": InferErrorResponse}},
)
def infer(
    request: Request,
    body: InferRequestBody,
    svc: Annotated[ModelService, Depends(get_model_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    horizon: Annotated[int | None, Query(ge=1)] = None,
    _credentials: Annotated[HTTPAuthorizationCredentials | None, Security(auth_scheme)] = None,
) -> InferSuccessResponse:
    verify_auth_token(settings, request)
    if not svc.is_ready:
        if svc.load_error is None:
            raise ModelNotReadyError("Model is still loading; retry after GET /ready reports ready.")
        raise ModelNotReadyError("Model failed to load", details={"load_error": svc.load_error})
    h = horizon if horizon is not None else settings.default_horizon
    predictions = svc.infer(body, h)
    request_id = getattr(request.state, "request_id", "")
    return InferSuccessResponse(
        request_id=request_id,
        model=body.model,
        horizon=h,
        predictions=predictions,
    )
