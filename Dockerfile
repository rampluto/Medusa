# HF Space root-level Dockerfile — targets port 7860 (HF default).
# This file lives at envs/medusa_env/Dockerfile and is the file
# HF Spaces uses when deploying a Docker Space from this directory.

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv --no-cache-dir

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install all dependencies including openenv-core + pandas + numpy
RUN uv pip install --system --no-cache \
    "openenv-core[core]>=0.2.2" \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pandas \
    numpy \
    websockets

# Install the medusa package itself (so medusa_env.* imports resolve)
RUN uv pip install --system --no-cache -e .

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# PYTHONPATH so imports resolve correctly when running from /app/env
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# Health check on HF port
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run on port 7860 — HF Space requirement
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
