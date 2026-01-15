# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS dependencies (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
# Copy first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# ---- Application code ----
COPY . /app

# Make src/ importable
ENV PYTHONPATH=/app

# ---- Entrypoint ----
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# Run as non-root (good VM hygiene)
RUN useradd -m appuser
USER appuser

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["pipeline"]
