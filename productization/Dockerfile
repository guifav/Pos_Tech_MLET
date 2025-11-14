# syntax=docker/dockerfile:1.7
# Base image with CUDA, cuDNN, and NVIDIA Container Toolkit support for GPU workloads
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install Python 3.13 and system dependencies required for scientific stacks and build tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        build-essential \
        git \
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        liblzma-dev \
        wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 && \
    python3.13 -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# Install nvidia-container-toolkit validation utility to fail fast if GPUs are unavailable
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir nvidia-ml-py3

WORKDIR /app

# Copy dependency definitions first to leverage Docker layer caching
COPY src/pyproject.toml ./pyproject.toml
COPY src/README.md ./README.md
COPY src/ ./

# Install project (CPU/GPU aware libs pinned in pyproject)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir .

# Create non-root user for runtime safety
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

# Validate GPU visibility at container startup; fail early if not available
HEALTHCHECK --interval=1m --timeout=10s --retries=3 CMD nvidia-smi || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
