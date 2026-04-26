# Hugging Face Space (Docker) — Medusa + Studio UI + GRPO Hub predictor.
# `hf_space/Dockerfile` is a symlink to this file (see `hf_space/README.md`).
#
# Requirements:
#   - Space: SDK = Docker, enable a GPU (T4+). CPU-only is not suitable for
#     Qwen2.5-3B + LoRA in this setup.
#   - Optional: add HF_TOKEN as a *Secret* if Qwen2.5-3B (gated) or the
#     adapter download fails, or the adapter is private.
#
# After deploy, open: https://<space>.hf.space/medusa/studio
# Select agent "grpo_trained" to use the remote adapter.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade --no-cache-dir pip uv

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

# Web server + openenv
RUN uv pip install --system --no-cache \
    "openenv-core[core]>=0.2.2" \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pandas \
    numpy \
    websockets

# PyTorch (CUDA 12.1) — install before peft/bnb so wheels stay aligned
RUN uv pip install --system --no-cache \
    "torch==2.4.*" \
    --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system --no-cache \
    "transformers>=4.45,<5" \
    "peft>=0.12" \
    "accelerate>=0.34" \
    "bitsandbytes>=0.43" \
    "sentencepiece>=0.2" \
    "huggingface_hub>=0.25"

# Editable medusa (medusa_env.*, server.*)
RUN uv pip install --system --no-cache -e .

ENV PORT=7860
EXPOSE 7860

ENV PYTHONPATH="/app/env"
ENV ENABLE_WEB_INTERFACE=true

# GRPO UI agent — defaults match eval / training stack; override in Space UI if needed
ENV MEDUSA_GRPO_PREDICTOR="trainer.grpo_predictor_hub:predict"
ENV BASE_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
ENV GRPO_MODEL_ID="anubhavkamal/medusa-qwen-grpo"
ENV MEDUSA_GRPO_LOAD_IN_4BIT="0"
ENV MEDUSA_GRPO_NO_MERGE="0"
ENV MEDUSA_GRPO_MAX_NEW_TOKENS="192"
ENV MEDUSA_GRPO_TEMPERATURE="0.1"
# T4 16GB: set MEDUSA_GRPO_LOAD_IN_4BIT=1 in Space variables (or rebuild with 1 here)

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
