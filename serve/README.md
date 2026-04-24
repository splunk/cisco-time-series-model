# Self-Hosting the Cisco Deep Time Series Model (CDTSM)

A self-hosted [FastAPI](https://fastapi.tiangolo.com/) inference server that loads the
[Cisco Deep Time Series Model](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0)
from Hugging Face and exposes an **AITK-compatible** JSON API for multiresolution forecasting.

This directory is part of the
[Cisco Deep Time Series Model](https://github.com/splunk/cisco-time-series-model) repository.
See the [root README](https://github.com/splunk/cisco-time-series-model/blob/main/README.md) for model details, benchmarks, and the `cisco-tsm` PyPI package.

## Model version note

The PyPI package [`cisco-tsm`](https://pypi.org/project/cisco-tsm/) ships the **CDTSM 1.0**
modeling code and matches the **`cisco-ai/cisco-time-series-model-1.0`** weights
(`num_layers=25`, extended quantile head).

## Prerequisites

- **Python 3.11** (required for both process-based and Docker-based hosting)
- **Docker** and **Docker Compose** (for Docker-based hosting only)
- **NVIDIA driver + nvidia-container-toolkit** (for GPU Docker images only)

## Quick Start

### 1. Configure environment

Create a `.env` file from the provided example and fill in the required values:

```bash
cd serve/
cp .env-example .env
```

Edit `.env` and set `CDTSM_AUTH_TOKEN` (required). All other variables have sensible defaults.
The `.env` file is git-ignored and will not be committed.
See [Environment Variables](#environment-variables) for the full list.

### 2a. Process-based (Makefile)

Use the Makefile for a quick local setup. The HTTP server starts immediately while the
model downloads and loads in the background.

**CPU:**

```bash
export CDTSM_AUTH_TOKEN=<your-token>   # or set in .env
make install-dev    # creates .venv, installs dependencies
make model-up       # starts the server on port 8080
```

**GPU (NVIDIA CUDA):**

```bash
export CDTSM_AUTH_TOKEN=<your-token>
make install-dev-gpu   # creates .venv, installs CUDA PyTorch + dependencies
make model-up          # starts the server on port 8080
```

In another terminal, wait until weights are downloaded and loaded:

```bash
make wait-ready      # polls GET /ready until the model is ready
```

### 2b. Docker

Docker Compose reads `.env` automatically via the `env_file` directive.
Model weights are cached in a Docker named volume (`hf_cache`) so they persist across
container restarts without re-downloading.

**CPU:**

```bash
docker compose up --build
```

**GPU (NVIDIA CUDA):**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

> **Note:** The container starts as root, fixes volume permissions via
> `entrypoint.sh`, then drops to a non-root `appuser` before running the server.

### 3. Verify

```bash
# Liveness
curl http://localhost:8080/health

# Readiness (returns 503 until the model is loaded)
curl http://localhost:8080/ready
```

### Environment Variables

All variables are prefixed with `CDTSM_`. Set them in `.env` or export before running.

| Variable | Default | Description |
|---|---|---|
| `CDTSM_AUTH_TOKEN` | *(required)* | Bearer token for API authentication |
| `CDTSM_HOST` | `0.0.0.0` | Server bind address |
| `CDTSM_PORT` | `8080` | Server port |
| `CDTSM_HF_REPO_ID` | `cisco-ai/cisco-time-series-model-1.0` | Hugging Face model repository |
| `CDTSM_TORCH_BACKEND` | `cpu` (CPU image) / `auto` (GPU image) | PyTorch backend: `cpu`, `gpu`, or `auto` |
| `CDTSM_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CDTSM_HF_INSECURE_SSL` | `false` | Disable TLS verification for Hub downloads (corporate proxy workaround) |
| `CDTSM_HF_SSL_CA_BUNDLE` | *(unset)* | Path to custom PEM CA bundle for Hugging Face HTTPS |
| `CDTSM_SERVING_BACKEND` | `native` | Serving backend. Only `native` (in-process) is supported — external runtimes such as vLLM, TGI, or Triton are not compatible with CDTSM. |

## API Reference

### `GET /health`

Liveness probe — returns **200** if the process is running.

### `GET /ready`

Readiness probe — returns **200** once the model is loaded and ready for inference.
Returns **503** with `model_load` status while the model is downloading or initializing.

### `POST /cdtsm/v1/ai/infer?horizon=128`

Multiresolution time series forecasting.

- **Headers**: `Authorization: Bearer <CDTSM_AUTH_TOKEN>` (required), `Content-Type: application/json`, optional `request_id` (echoed back).

**Quantile labels**

`metadata.quantiles` in the request selects which outputs the server emits; the
same labels appear as keys under `predictions[i].quantiles` in the response.
The supported set is discrete (not every integer between 1 and 99):

| Request label | Meaning                 |
|---------------|-------------------------|
| `mean`        | Point forecast (mean)   |
| `p1`, `p5`, `p10`, `p20`, `p25` | 0.01, 0.05,  0.10,  0.20, 0.25 |
| `p30`, `p40`, `p50`, `p60`, `p70` | 0.30, 0.40, 0.50, 0.60, 0.70 |
| `p75`, `p80`, `p90`, `p95`, `p99` | 0.75, 0.80, 0.90, 0.95, 0.99 |


**Request body:**

```json
{
  "payload": [
    {
      "coarse_ctx": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      "fine_ctx": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    }
  ],
  "model": "CDTSM",
  "metadata": {
    "quantiles": ["mean", "p40", "p90"]
  }
}
```

**Success response (200):** keys under `predictions[i].quantiles` mirror the
non-`mean` labels requested above (`p40`, `p90`). `mean` is always returned
under `predictions[i].mean`.

```json
{
  "request_id": "demo-1",
  "model": "CDTSM",
  "horizon": 128,
  "predictions": [
    {
      "mean": [ ... ],
      "quantiles": {
        "p40": [ ... ],
        "p90": [ ... ]
      }
    }
  ]
}
```

## Example `curl`

```bash
curl -s -X POST 'http://localhost:8080/cdtsm/v1/ai/infer?horizon=128' \
  -H 'Content-Type: application/json' \
  -H 'request_id: demo-1' \
  -H "Authorization: Bearer $CDTSM_AUTH_TOKEN" \
  -d '{
    "payload": [
      {
        "coarse_ctx": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "fine_ctx":   [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
      }
    ],
    "model": "CDTSM",
    "metadata": { "quantiles": ["mean", "p40", "p90"] }
  }'
```

### Interactive API explorer

FastAPI auto-generates browser-based API docs, which are usually the fastest way to
build a request without crafting JSON by hand:

- **Swagger UI** (`Try it out` button): [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc**: [http://localhost:8080/redoc](http://localhost:8080/redoc)
- **OpenAPI schema**: [http://localhost:8080/openapi.json](http://localhost:8080/openapi.json)

In Swagger UI, click **Authorize** and paste your `CDTSM_AUTH_TOKEN` (or the full
`Bearer <token>` header depending on how your browser extension handles it) before
invoking `POST /cdtsm/v1/ai/infer`.

## Running Tests

```bash
make install-dev   # if not already done
make test          # runs pytest
```

To run the full model-load integration test (downloads the model from Hugging Face):

```bash
RUN_MODEL_LOAD=1 make test
```

## License

This project is licensed under the Apache License 2.0. See the root repository for the full license text.
