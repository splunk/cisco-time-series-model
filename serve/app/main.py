from __future__ import annotations

from pathlib import Path

# HF_* env vars must exist before huggingface_hub imports (Pydantic .env is too late).
try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass

import asyncio
import uuid
from contextlib import asynccontextmanager
from math import isfinite

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.hf_client import configure_huggingface_http
from app.core.exceptions import CDTSMServiceError
from app.core.logging import get_logger, setup_logging
from app.models.schemas import ErrorDetail, InferErrorResponse
from app.services.model_service import get_model_service, get_v15_model_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info(
        "Starting CDTSM host",
        extra={"structured": {"service": settings.service_name}},
    )
    configure_huggingface_http(settings)
    svc = get_model_service()
    load_tasks = [asyncio.create_task(asyncio.to_thread(svc.load))]
    if settings.v15_checkpoint_path is not None:
        v15 = get_v15_model_service()
        load_tasks.append(asyncio.create_task(asyncio.to_thread(v15.load)))
    logger.info(
        "Background model load started (/health up; /ready until load completes)",
        extra={"structured": {"service": settings.service_name}},
    )
    try:
        yield
    finally:
        for task in load_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*load_tasks, return_exceptions=True)
        logger.info("Shutdown complete")


app = FastAPI(title="CDTSM Inference Host", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("request_id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["request_id"] = rid
    return response


@app.exception_handler(CDTSMServiceError)
async def cdtsm_service_error_handler(request: Request, exc: CDTSMServiceError) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) or ""
    body = InferErrorResponse(
        request_id=rid,
        error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) or ""
    errs = exc.errors()
    message = errs[0].get("msg", "validation error") if errs else "validation error"
    body = InferErrorResponse(
        request_id=rid,
        error=ErrorDetail(
            code="validation_error",
            message=message,
            details={"errors": errs},
        ),
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            body.model_dump(),
            custom_encoder={ValueError: str, float: lambda value: value if isfinite(value) else None},
        ),
    )
