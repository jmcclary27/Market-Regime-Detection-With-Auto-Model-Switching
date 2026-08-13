# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    RUFF_CACHE_DIR=/tmp/ruff_cache \
    PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache" \
    MYPY_CACHE_DIR=/tmp/mypy_cache

# -------------------------
# System deps
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    awscli \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# App setup
# -------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy repo
COPY . /app

# -------------------------
# Runtime user + caches
# -------------------------
RUN useradd -m appuser \
 && chown -R appuser:appuser /app \
 && mkdir -p /tmp/ruff_cache /tmp/pytest_cache /tmp/mypy_cache \
 && chown -R appuser:appuser /tmp/ruff_cache /tmp/pytest_cache /tmp/mypy_cache

USER appuser

# -------------------------
# Entrypoint
# -------------------------
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
CMD ["demo"]
