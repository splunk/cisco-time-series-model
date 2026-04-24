#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3.11}"
if ! command -v "$PY" &>/dev/null; then
  PY="python3"
fi
VENV="${VENV:-.venv}"
if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -U pip
pip install -r requirements-dev.txt
export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())" 2>/dev/null || true)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CDTSM_HOST="${CDTSM_HOST:-0.0.0.0}"
export CDTSM_PORT="${CDTSM_PORT:-8080}"
# Optional: CDTSM_TORCH_BACKEND=cpu|gpu|auto (install CUDA PyTorch from pytorch.org for GPU)
exec uvicorn app.main:app --host "$CDTSM_HOST" --port "$CDTSM_PORT"
