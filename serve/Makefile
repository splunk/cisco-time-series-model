.PHONY: venv install-dev install-dev-gpu model-up wait-ready test

PYTHON ?= python3.11
VENV ?= .venv
TORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu124
REQUIRED_PYTHON_VERSION := 3.11

define check_python_version
	@PY_VER=$$($(1) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null); \
	if [ "$$PY_VER" != "$(REQUIRED_PYTHON_VERSION)" ]; then \
		echo "ERROR: Python $(REQUIRED_PYTHON_VERSION) required, found $$PY_VER ($(1))"; \
		echo "Install Python $(REQUIRED_PYTHON_VERSION) and retry, or set PYTHON=python$(REQUIRED_PYTHON_VERSION)"; \
		exit 1; \
	fi
endef

venv:
	$(call check_python_version,$(PYTHON))
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install -U pip

install-dev: venv
	./$(VENV)/bin/pip install -r requirements-dev.txt

install-dev-gpu: venv
	./$(VENV)/bin/pip install torch --index-url $(TORCH_CUDA_INDEX)
	./$(VENV)/bin/pip install -r requirements-dev.txt
	./$(VENV)/bin/pip install --force-reinstall torch --index-url $(TORCH_CUDA_INDEX)

# HTTP listens immediately; model loads in background (HF download + torch.load).
# SSL_CERT_FILE: cert bundle for tools; HF_HUB_DISABLE_XET defaults to 1 (override with HF_HUB_DISABLE_XET=0).
model-up:
	$(call check_python_version,$(VENV)/bin/python)
	@if [ -z "$${CDTSM_AUTH_TOKEN}" ]; then echo "ERROR: CDTSM_AUTH_TOKEN is not set. Export it before running model-up."; exit 1; fi
	@if [ ! -x "$(VENV)/bin/uvicorn" ]; then $(MAKE) install-dev; fi
	@echo "CDTSM host starting. Poll: make wait-ready  or  GET /ready"
	@CERT=$$($(VENV)/bin/python -c "import certifi; print(certifi.where())" 2>/dev/null); \
	export SSL_CERT_FILE="$$CERT"; \
	export REQUESTS_CA_BUNDLE="$$CERT"; \
	export HF_HUB_DISABLE_XET=$${HF_HUB_DISABLE_XET:-1}; \
	$(VENV)/bin/uvicorn app.main:app \
		--host "$${CDTSM_HOST:-0.0.0.0}" \
		--port "$${CDTSM_PORT:-8080}"

wait-ready:
	@HOST=$${CDTSM_HOST:-127.0.0.1}; PORT=$${CDTSM_PORT:-8080}; MAX=$${WAIT_MAX_TRIES:-30}; \
	for i in $$(seq 1 $$MAX); do \
	  body=$$(curl -s "http://$$HOST:$$PORT/ready" 2>/dev/null) || true; \
	  echo "$$body" | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"' && echo "Inference ready at http://$$HOST:$$PORT" && exit 0; \
	  echo "[$$i/$$MAX] not ready yet..."; sleep 10; \
	done; echo "Timed out; check server logs and GET /ready load_error"; exit 1

test:
	./$(VENV)/bin/pytest -q
