"""Isolate upstream Cisco TSM construction behind a small factory."""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache

import torch
from cisco_tsm import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams

from app.core.config import Settings
from app.core.constants import CDTSM_DEFAULT_QUANTILES, CDTSM_NUM_LAYERS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def forecast_supports_restrict_quantiles() -> bool:
    """Return True when ``CiscoTsmMR.forecast`` accepts ``restrict_quantiles``."""
    supported = "restrict_quantiles" in inspect.signature(CiscoTsmMR.forecast).parameters
    if not supported:
        logger.warning("Installed cisco-tsm does not accept restrict_quantiles; quantile truncation disabled")
    return supported


def resolve_inference_backend(settings: Settings) -> str:
    """Map config to TimesFm backend ('cpu' | 'gpu')."""
    if settings.torch_backend == "cpu":
        return "cpu"
    if settings.torch_backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CDTSM_TORCH_BACKEND=gpu but CUDA is not available to PyTorch. "
                "Use CDTSM_TORCH_BACKEND=cpu for CPU-only hosts, or auto for best effort, "
                "or install a CUDA-enabled PyTorch build and NVIDIA drivers (see README)."
            )
        return "gpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_cisco_tsm_model(settings: Settings) -> tuple[CiscoTsmMR, str]:
    """Build the CiscoTsmMR model; weights are fetched by the upstream loader."""
    backend = resolve_inference_backend(settings)
    logger.info(
        "Initializing CiscoTsmMR",
        extra={"structured": {"hf_repo": settings.hf_repo_id, "backend": backend, "num_layers": CDTSM_NUM_LAYERS}},
    )
    hparams = TimesFmHparams(
        num_layers=CDTSM_NUM_LAYERS,
        use_positional_embedding=False,
        backend=backend,
        quantiles=CDTSM_DEFAULT_QUANTILES,
    )
    ckpt = TimesFmCheckpoint(huggingface_repo_id=settings.hf_repo_id)
    model = CiscoTsmMR(hparams=hparams, checkpoint=ckpt)
    return model, backend
