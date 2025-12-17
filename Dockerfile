ARG BASE_IMAGE=python:3.10.13-slim
FROM ${BASE_IMAGE} AS base

ARG TORCH_WHL_CHANNEL=cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_INDEX_URL=https://download.pytorch.org/whl/${TORCH_WHL_CHANNEL} \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6 \
        libfreetype6-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app

WORKDIR /app/1.0-preview

EXPOSE 8000

ENV CPU_FORECAST_SERVER_PORT=8000

CMD ["python", "cpu_forecast_server.py"]
