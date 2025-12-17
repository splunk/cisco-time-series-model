# Docker Usage Guide

This guide explains how to build and run the Docker image that serves the Flask CPU forecast app defined in `1.0-preview/cpu_forecast_server.py`.

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux) 24.0+.
- Internet connectivity to download Python dependencies and the Cisco TSM checkpoint on first launch.

## Build the Image

Run from the repository root:

```bash
docker build -t cisco-cpu-forecast:latest .
```

This copies the repo, installs `requirements.txt`, and sets `cpu_forecast_server.py` as the default entrypoint listening on port `8000`. The Docker image uses Python 3.10.13 to match the verified runtime declared in `.python-version`.

### Build with GPU Wheels (optional)

The Dockerfile defaults to CPU-only PyTorch wheels by pointing `PIP_INDEX_URL` at `https://download.pytorch.org/whl/cpu`. To target NVIDIA wheels, pass the CUDA channel via the `TORCH_WHL_CHANNEL` build argument, e.g.:

```bash
# CUDA 12.1 wheels
docker build --build-arg TORCH_WHL_CHANNEL=cu121 -t cisco-cpu-forecast:gpu .
```

The final `PIP_INDEX_URL` becomes `https://download.pytorch.org/whl/<channel>`. Use any channel published by PyTorch (`cu118`, `cu121`, etc.). When no argument is provided the build remains CPU-only.

### Use the Published GitHub Container Registry Image

If you prefer not to build locally, every push to `main` automatically publishes an image to the GitHub Container Registry (GHCR). Authenticate with GHCR (a classic PAT with the `read:packages` scope or the standard `GITHUB_TOKEN` in CI) and pull:

```bash
# replace <owner> with the GitHub org/user that hosts this repo
docker pull ghcr.io/<owner>/cisco-cpu-forecast:cpu-latest   # CPU build (default)
docker pull ghcr.io/<owner>/cisco-cpu-forecast:gpu-latest   # GPU build (CUDA 12.1 wheels)
docker run --rm -it -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:cpu-latest
```

> CI jobs also upload compressed Docker images (`cpu-forecast-server-cpu-image.tar.gz` and `cpu-forecast-server-gpu-image.tar.gz`) as GitHub Action artifacts. Download them from the workflow run summary, decompress, and load with `docker load -i <artifact>`.

## Run the Container

The container exposes port `8000` via the `CPU_FORECAST_SERVER_PORT` environment variable. Map it to your host port with `-p 8000:8000`.

Choose the image that matches your hardware:

- **CPU** (portable, no GPU dependencies): `ghcr.io/<owner>/cisco-cpu-forecast:cpu-latest`
- **GPU** (requires NVIDIA Container Toolkit + CUDA-capable hardware): `ghcr.io/<owner>/cisco-cpu-forecast:gpu-latest`

### macOS / Linux (bash/zsh)

```bash
# CPU example
docker run --rm -it -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:cpu-latest

# GPU example (Docker host must expose GPUs, e.g., --gpus all on Docker CLI)
docker run --rm -it --gpus all -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:gpu-latest
```

### Windows PowerShell

```powershell
# CPU
docker run --rm -it -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:cpu-latest

# GPU
docker run --rm -it --gpus all -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:gpu-latest
```

> The commands are identical across platforms. Ensure Docker Desktop is running on Windows/macOS. Linux requires the Docker daemon to be active.

## Validate the Server

1. Wait until the container logs show Flask listening on `0.0.0.0:8000`.
2. Open your browser to http://localhost:8000 to load the web UI. Submit the default sample to confirm graphs render.
3. (Optional) Use `curl` to exercise the API endpoint:

   ```bash
   curl -X POST http://localhost:8000/api/forecast \
     -H "Content-Type: application/json" \
     -d '{"use_sample": true, "horizon": 128}'
   ```

   A PNG response (`image/png`) indicates the service is healthy.

4. Stop the container with `Ctrl+C` (Linux/macOS) or `Ctrl+Break` (Windows PowerShell), or by running `docker stop <container_id>` from another terminal.

## Troubleshooting

- **Port already in use:** Change the host side of the mapping, e.g., `-p 8080:8000`, then visit http://localhost:8080.
- **Slow first request:** The model downloads weights on first use; subsequent runs are faster because the checkpoint remains in the Docker layer cache.
- **GPU acceleration:** Build with `--build-arg TORCH_WHL_CHANNEL=<cu version>` and run on a host with the NVIDIA Container Toolkit installed so Docker can expose the GPU to the container. Without the arg, the image uses CPU wheels.
- **CPU vs GPU tags:** `*-cpu-*` tags work everywhere; `*-gpu-*` tags assume CUDA-compatible hardware. Pull the tag that matches your environment to avoid unnecessary dependencies.
