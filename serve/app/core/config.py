from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TorchBackendMode = Literal["cpu", "gpu", "auto"]


def _coerce_bool(v: object) -> bool:
    """Coerce env-var strings ('1', 'true', 'yes', 'on') to bool."""
    if v is None or v is False:
        return False
    if v is True:
        return True
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CDTSM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Service                                                            #
    # ------------------------------------------------------------------ #
    service_name: str = "cdtsm-inference-host"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Hugging Face                                                       #
    # ------------------------------------------------------------------ #
    hf_repo_id: str = Field(
        default="cisco-ai/cisco-time-series-model-1.0",
        description="Hugging Face repo id for torch checkpoint",
    )
    hf_ssl_ca_bundle: str | None = Field(
        default=None,
        description="Path to PEM CA bundle for Hugging Face HTTPS",
    )
    hf_insecure_ssl: bool = Field(
        default=False,
        description="If true, disable TLS verification for Hub calls (dev/lab only)",
    )
    hf_hub_download_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=86400,
        description="Hub streaming read timeout in seconds",
    )

    # ------------------------------------------------------------------ #
    # Model                                                              #
    # ------------------------------------------------------------------ #
    torch_backend: TorchBackendMode = Field(
        default="auto",
        description="cpu: force CPU | gpu: require CUDA | auto: GPU if available else CPU",
    )
    serving_backend: str = Field(
        default="native",
        description=(
            "Serving backend. Only ``native`` (in-process CiscoTsmMR) is "
            "supported; external runtimes such as vLLM, TGI, or Triton are "
            "not compatible with CDTSM and will be rejected at load time."
        ),
    )

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    infer_batch_size: int = Field(
        default=8,
        ge=1,
        description=(
            "Batch size passed to the model's forecast() call. "
            "On higher-memory GPUs, values well above the default (e.g. >256) "
            "can meaningfully improve throughput; no upper bound is enforced."
        ),
    )
    default_horizon: int = Field(default=128, ge=1)

    # ------------------------------------------------------------------ #
    # Auth                                                               #
    # ------------------------------------------------------------------ #
    auth_token: str = Field(
        description="Required. Callers must send Authorization: Bearer <CDTSM_AUTH_TOKEN> on /infer",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                         #
    # ------------------------------------------------------------------ #
    @field_validator("hf_insecure_ssl", mode="before")
    @classmethod
    def coerce_hf_insecure_ssl(cls, v: object) -> bool:
        return _coerce_bool(v)

    @field_validator("torch_backend", mode="before")
    @classmethod
    def normalize_torch_backend(cls, v: object) -> str:
        if v is None:
            return "auto"
        s = str(v).strip().lower()
        if s not in ("cpu", "gpu", "auto"):
            raise ValueError("torch_backend must be one of: cpu, gpu, auto")
        return s

    @field_validator("serving_backend", mode="before")
    @classmethod
    def normalize_serving_backend(cls, v: object) -> str:
        if v is None:
            return "native"
        s = str(v).strip().lower()
        if not s:
            raise ValueError("serving_backend must not be empty")
        return s


@lru_cache
def get_settings() -> Settings:
    return Settings()
