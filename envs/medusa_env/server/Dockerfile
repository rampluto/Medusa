# Multi-stage build — mirrors the kernrl Dockerfile pattern.
# Works both in-repo (openenv-core from local src/) and standalone (from pip).

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install dependencies (pandas, numpy, fastapi, uvicorn, pydantic, etc.)
# pyproject.toml declares all required deps including pandas + numpy
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pandas \
    numpy \
    websockets

# Install openenv-core from the local src/ if present (in-repo build),
# otherwise fall back to pip (standalone build)
RUN if [ -f /app/env/src/openenv/__init__.py ]; then \
        uv pip install --system --no-cache -e /app/env; \
    else \
        uv pip install --system --no-cache openenv-core; \
    fi

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages and environment code from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/env /app/env

# PYTHONPATH so `from envs.medusa_env import ...` and
# `from openenv.core import ...` both resolve correctly
ENV PYTHONPATH="/app/env:/app/env/src:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run the FastAPI server from the repo root so `envs.medusa_env.*` imports work
CMD ["uvicorn", "envs.medusa_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
