# =============================================================================
# ESG Data Foundry — Dockerfile (Python 3.12 slim)
# Multi-stage build: builder installs wheels, runtime image stays lean.
# =============================================================================

ARG PYTHON_VERSION=3.12

# ── Stage 1: Build wheels ────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# System deps needed to compile psycopg2-binary, cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

# Install libpq for psycopg2 at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application source
COPY agents/    ./agents/
COPY config/    ./config/
COPY db/        ./db/
COPY pipeline/  ./pipeline/
COPY utils/     ./utils/
COPY init/      ./init/
COPY main.py    ./main.py

# Output directory (mounted as volume in docker-compose)
RUN mkdir -p /app/output /app/data

# Non-root user for security
RUN groupadd -r foundry && useradd -r -g foundry foundry && \
    chown -R foundry:foundry /app
USER foundry

ENTRYPOINT ["python", "main.py"]
